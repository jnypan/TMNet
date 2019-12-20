from __future__ import print_function
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import sys
sys.path.append('./auxiliary/')
from dataset import *
from model import *
from utils import *
from ply import *
import os
import json
import datetime
import visdom
import scipy.io as sio
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=12)
parser.add_argument('--nepoch', type=int, default=120, help='number of epochs to train for')
parser.add_argument('--epoch_decay',type=int,default=100,help='epoch to decay lr')
parser.add_argument('--model', type=str,default='./log/SVR_subnet2/network.pth',help='model path from the trained subnet2')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT point cloud')
parser.add_argument('--num_vertices', type=int, default=2562)
parser.add_argument('--env', type=str, default="SVR_subnet3", help='visdom env')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--tau_decay', type=float, default=2.0)
parser.add_argument('--lambda_boundary', type=float, default=0.5)
parser.add_argument('--lambda_displace', type=float, default=0.2)
parser.add_argument('--pool',type=str,default='max',help='max or mean or sum' )
parser.add_argument('--manualSeed',type=int,default=6185)
opt = parser.parse_args()
print(opt)

sys.path.append("./extension/")
import dist_chamfer as ext
distChamfer = ext.chamferDist()

server = 'http://localhost/'
vis = visdom.Visdom(server=server, port=8888, env=opt.env, use_incoming_socket=False)
now = datetime.datetime.now()
save_path = opt.env
dir_name = os.path.join('./log', save_path)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
logname = os.path.join(dir_name, 'log.txt')
blue = lambda x: '\033[94m' + x + '\033[0m'
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = ShapeNet(npoints=opt.num_points, SVR=True, normal=True,train=True,class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=False,class_choice='chair')
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
print('training set', len(dataset.datapath))
print('testing set', len(dataset_test.datapath))
len_dataset = len(dataset)

name = 'sphere' + str(opt.num_vertices) + '.mat'
mesh = sio.loadmat('./data/' + name)
faces = np.array(mesh['f'])
faces_cuda = torch.from_numpy(faces.astype(int)).type(torch.cuda.LongTensor)
vertices_sphere = np.array(mesh['v'])
vertices_sphere = (torch.cuda.FloatTensor(vertices_sphere)).transpose(0, 1).contiguous()
vertices_sphere = vertices_sphere.contiguous().unsqueeze(0)
edge_cuda = get_edges(faces)

network = SVR_TMNet()
network.apply(weights_init)
network.cuda()
if opt.model != '':
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(opt.model).items() if (k in model_dict)}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    print(" Previous weight loaded ")
print(network)
network.cuda()

lrate = opt.lr
optimizer = optim.Adam(network.refine.parameters(), lr=lrate)

train_CDs_stage3_loss = AverageValueMeter()
val_CDs_stage3_loss = AverageValueMeter()
train_boundary_loss = AverageValueMeter()
val_boundary_loss = AverageValueMeter()
train_displace_loss = AverageValueMeter()
val_displace_loss = AverageValueMeter()

with open(logname, 'a') as f:
    f.write(str(network) + '\n')

train_CDs_stage3_curve = []
val_CDs_stage3_curve = []
train_boundary_curve = []
val_boundary_curve = []
train_displace_curve = []
val_displace_curve = []


for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_CDs_stage3_loss.reset()
    train_boundary_loss.reset()
    train_displace_loss.reset()

    network.eval()
    network.refine.train()

    if epoch == opt.epoch_decay:
        optimizer = optim.Adam(network.refine.parameters(), lr=lrate / 10.0)

    for i, data in enumerate(dataloader, 0):
        optimizer.zero_grad()
        img, points, normals, name, cat = data
        img = img.cuda()
        points = points.cuda()
        normals = normals.cuda()
        choice = np.random.choice(points.size(1), opt.num_vertices, replace=False)
        points_choice = points[:, choice, :].contiguous()
        vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                 vertices_sphere.size(2)).contiguous())

        with torch.no_grad():
            pointsRec = network(img, vertices_input,mode='deform1')
            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            error_stage1 = network(img, pointsRec_samples.detach().transpose(1, 2),mode='estimate')
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(img.size(0), faces_cuda.size(0), faces_cuda.size(1))
            faces_cuda_bn = prune(faces_cuda_bn.detach(), error_stage1.detach(), opt.tau, index, opt.pool)

            pointsRec2 = network(img, pointsRec, mode='deform2')
            pointsRec2_samples, index = samples_random(faces_cuda_bn.detach(), pointsRec2, opt.num_points)
            error = network(img, (pointsRec2_samples.detach()).transpose(1, 2), mode='estimate2')
            faces_cuda_bn = faces_cuda_bn.clone()
            faces_cuda_bn = prune(faces_cuda_bn, error.detach(), opt.tau/opt.tau_decay, index, opt.pool)

            indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0))).type(torch.cuda.LongTensor)

            pointsRec2_boundary, selected_pair_all, selected_pair_all_len = get_boundary_points_bn(faces_cuda_bn,pointsRec2)
            vector1 = (pointsRec2_boundary[:, :, 1] - pointsRec2_boundary[:, :, 0])
            vector2 = (pointsRec2_boundary[:, :, 2] - pointsRec2_boundary[:, :, 0])
            vector1 = vector1 / (torch.norm((vector1 + 1e-6), dim=2)).unsqueeze(2)
            vector2 = vector2 / (torch.norm((vector2 + 1e-6), dim=2)).unsqueeze(2)
            vector1 = vector1.transpose(2,1).detach()
            vector2 = vector2.transpose(2,1).detach()

        # refine the boundary and get the displace loss
        if pointsRec2_boundary.shape[1] != 0:
            pointsRec3_boundary = network(img, pointsRec2_boundary[:, :, 0], vector1, vector2, mode='refine')
        else:
            pointsRec3_boundary = pointsRec2_boundary[:, :, 0]
        displace_loss = pointsRec3_boundary - pointsRec2_boundary[:, :, 0]
        displace_loss = torch.mean(torch.abs(displace_loss))

        # get the final refined mesh and cd loss
        pointsRec3_set = []
        for ibatch in torch.arange(0, img.shape[0]):
            length = selected_pair_all_len[ibatch]
            if length != 0:
                index_bp = selected_pair_all[ibatch][:, 0][:length]
                prb_final = pointsRec3_boundary[ibatch][:length]
                pr = pointsRec2[ibatch]
                index_bp = index_bp.view(index_bp.shape[0], -1).expand([index_bp.shape[0], 3])
                pr_final = pr.scatter(dim=0, index=index_bp, source=prb_final)
                pointsRec3_set.append(pr_final)
            else:
                pr = pointsRec2[ibatch]
                pr_final = pr
                pointsRec3_set.append(pr_final)
        pointsRec3 = torch.stack(pointsRec3_set, 0)
        pointsRec3_samples, _ = samples_random(faces_cuda_bn, pointsRec3, opt.num_points)
        dist13_samples, dist23_samples, _, _ = distChamfer(points, pointsRec3_samples)
        cds_stage3 = (torch.mean(dist13_samples)) + (torch.mean(dist23_samples))

        # get the boundary loss
        points_select = pointsRec3.index_select(1, selected_pair_all.view(-1)).\
            view(pointsRec3.size(0) * selected_pair_all.size(0),selected_pair_all.size(1), selected_pair_all.size(2),
                 pointsRec3.size(2))
        points_select = points_select.index_select(0, indices)
        edge_a = points_select[:, :, 0] - points_select[:, :, 1]
        edge_b = points_select[:, :, 0] - points_select[:, :, 2]
        edge_a_norm = edge_a / (torch.norm((edge_a + 1e-6), dim=2)).unsqueeze(2)
        edge_b_norm = edge_b / (torch.norm((edge_b + 1e-6), dim=2)).unsqueeze(2)
        final = torch.abs(edge_a_norm + edge_b_norm).sum(2)
        loss_mask = (selected_pair_all.sum(2) != 0).type(torch.cuda.FloatTensor).detach()
        loss_boundary_final = (final * loss_mask).sum() / len(loss_mask.nonzero())

        loss_net = cds_stage3 + opt.lambda_boundary * loss_boundary_final + opt.lambda_displace * displace_loss
        loss_net.backward()
        train_CDs_stage3_loss.update(cds_stage3.item())
        train_boundary_loss.update(loss_boundary_final.item())
        train_displace_loss.update(displace_loss.item())
        optimizer.step()

        # VIZUALIZE
        if i % 50 <= 0:
            vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN"))
            vis.scatter(X=points_choice[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title="TRAIN_INPUT",
                            markersize=2,
                        ),
                        )
            vis.scatter(X=pointsRec3_samples[0].data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )
        print('[%d: %d/%d] train_cd_loss:  %f' % (epoch, i, len_dataset / opt.batchSize, cds_stage3.item()))

    train_CDs_stage3_curve.append(train_CDs_stage3_loss.avg)
    train_boundary_curve.append(train_boundary_loss.avg)
    train_displace_curve.append(train_displace_loss.avg)

    with torch.no_grad():
        # VALIDATION
        val_CDs_stage3_loss.reset()
        val_boundary_loss.reset()
        val_displace_loss.reset()

        for item in dataset_test.cat:
            dataset_test.perCatValueMeter[item].reset()

        network.eval()
        for i, data in enumerate(dataloader_test, 0):
            img, points, normals, name, cat = data
            img = img.cuda()
            points = points.cuda()
            normals = normals.cuda()
            choice = np.random.choice(points.size(1), opt.num_vertices, replace=False)
            points_choice = points[:, choice, :].contiguous()
            vertices_input = (vertices_sphere.expand(img.size(0), vertices_sphere.size(1),
                                                     vertices_sphere.size(2)).contiguous())
            pointsRec = network(img, vertices_input,mode='deform1')  # vertices_sphere 3*2562
            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            error = network(img, pointsRec_samples.detach().transpose(1, 2),mode='estimate')
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(img.size(0), faces_cuda.size(0), faces_cuda.size(1))
            faces_cuda_bn = prune(faces_cuda_bn, error, opt.tau, index, opt.pool)

            pointsRec2 = network(img, pointsRec, mode='deform2')
            pointsRec2_samples, index = samples_random(faces_cuda_bn, pointsRec2, opt.num_points)
            error = network(img, pointsRec2_samples.detach().transpose(1, 2), mode='estimate2')
            faces_cuda_bn = faces_cuda_bn.clone()
            faces_cuda_bn = prune(faces_cuda_bn, error, opt.tau/opt.tau_decay, index, opt.pool)

            indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0))).type(torch.cuda.LongTensor)

            pointsRec2_boundary, selected_pair_all, selected_pair_all_len = \
                get_boundary_points_bn(faces_cuda_bn, pointsRec2)
            vector1 = (pointsRec2_boundary[:, :, 1] - pointsRec2_boundary[:, :, 0])
            vector2 = (pointsRec2_boundary[:, :, 2] - pointsRec2_boundary[:, :, 0])
            vector1 = vector1 / (torch.norm((vector1 + 1e-6), dim=2)).unsqueeze(2)
            vector2 = vector2 / (torch.norm((vector2 + 1e-6), dim=2)).unsqueeze(2)
            vector1 = vector1.transpose(2, 1).detach()
            vector2 = vector2.transpose(2, 1).detach()

            # refine the boundary and get the displace loss
            if pointsRec2_boundary.shape[1] != 0:
                pointsRec3_boundary = network\
                    (img, pointsRec2_boundary[:, :, 0], vector1, vector2, mode='refine')
            else:
                pointsRec3_boundary = pointsRec2_boundary[:, :, 0]
            displace_loss = pointsRec3_boundary - pointsRec2_boundary[:, :, 0]
            displace_loss = torch.mean(torch.abs(displace_loss))

            # get the final refined mesh and cd loss
            pointsRec3_set = []
            for ibatch in torch.arange(0, img.shape[0]):
                length = selected_pair_all_len[ibatch]
                if length != 0:
                    index_bp = selected_pair_all[ibatch][:, 0][:length]
                    prb_final = pointsRec3_boundary[ibatch][:length]
                    pr = pointsRec2[ibatch]
                    index_bp = index_bp.view(index_bp.shape[0], -1).expand([index_bp.shape[0], 3])
                    pr_final = pr.scatter(dim=0, index=index_bp, source=prb_final)
                    pointsRec3_set.append(pr_final)
                else:
                    pr = pointsRec2[ibatch]
                    pr_final = pr
                    pointsRec3_set.append(pr_final)
            pointsRec3 = torch.stack(pointsRec3_set, 0)
            pointsRec3_samples, _ = samples_random (faces_cuda_bn, pointsRec3, opt.num_points)
            dist13_samples, dist23_samples, _, _ = distChamfer(points,pointsRec3_samples)
            cds_stage3 = (torch.mean(dist13_samples)) + (torch.mean(dist23_samples))

            # get the boundary loss
            points_select = pointsRec3.index_select(1, selected_pair_all.view(-1)). \
                view(pointsRec3.size(0) * selected_pair_all.size(0),
                     selected_pair_all.size(1), selected_pair_all.size(2),
                     pointsRec3.size(2))
            points_select = points_select.index_select(0, indices)
            edge_a = points_select[:, :, 0] - points_select[:, :, 1]
            edge_b = points_select[:, :, 0] - points_select[:, :, 2]
            edge_a_norm = edge_a / (torch.norm((edge_a + 1e-6), dim=2)).unsqueeze(2)
            edge_b_norm = edge_b / (torch.norm((edge_b + 1e-6), dim=2)).unsqueeze(2)
            final = torch.abs(edge_a_norm + edge_b_norm).sum(2)
            loss_mask = (selected_pair_all.sum(2) != 0).type(torch.cuda.FloatTensor).detach()
            loss_boundary_final = (final * loss_mask).sum() / len(loss_mask.nonzero())

            val_CDs_stage3_loss.update(cds_stage3.item())
            val_boundary_loss.update(loss_boundary_final.item())
            val_displace_loss.update(displace_loss.item())
            dataset_test.perCatValueMeter[cat[0]].update(cds_stage3.item())

            if i % 25 == 0:
                vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE TRAIN"))
                vis.scatter(X=points_choice[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsRec3_samples[0].data.cpu(),
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )

            print('[%d: %d/%d] val_cd_loss:  %f ' % (epoch, i, len(dataset_test)/opt.batchSize, cds_stage3.item()))

        val_CDs_stage3_curve.append(val_CDs_stage3_loss.avg)
        val_boundary_curve.append(val_boundary_loss.avg)
        val_displace_curve.append(val_displace_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_boundary_curve)), np.arange(len(val_boundary_curve)))),
             Y=np.log(np.column_stack((np.array(train_boundary_curve), np.array(val_boundary_curve)))),
             win='boundary_loss',
             opts=dict(title="boundary_loss", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_displace_curve)), np.arange(len(val_displace_curve)))),
             Y=np.log(np.column_stack((np.array(train_displace_curve), np.array(val_displace_curve)))),
             win='displace_loss',
             opts=dict(title="displace_loss", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_CDs_stage3_curve)), np.arange(len(val_CDs_stage3_curve)))),
             Y=np.log(np.column_stack((np.array(train_CDs_stage3_curve), np.array(val_CDs_stage3_curve)))),
             win='CDs_stage3',
             opts=dict(title="CDs_stage3", legend=["train", "val"], markersize=2, ), )

    log_table = {
        "train_cds_stage3_loss" : train_CDs_stage3_loss.avg,
        "val_cds_stage3_loss": val_CDs_stage3_loss.avg,
        "epoch": epoch,
        "lr": lrate,
    }

    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:  # open and append
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
