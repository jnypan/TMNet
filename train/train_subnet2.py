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
parser.add_argument('--epoch_decay',type=int,default=100, help='epoch to decay lr ')
parser.add_argument('--model', type=str,default='./log/SVR_subnet1/network.pth',help='model path from the trained subnet1')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT point cloud')
parser.add_argument('--num_vertices', type=int, default=2562)
parser.add_argument('--num_samples',type=int,default=2500, help='number of samples for error estimation')
parser.add_argument('--env', type=str, default="SVR_subnet2", help='visdom env')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--tau_decay', type=float, default=2.0)
parser.add_argument('--lambda_normal', type=float, default=5e-3)
parser.add_argument('--lambda_edge', type=float, default=0.05)
parser.add_argument('--lambda_smooth', type=float, default=2e-7)
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

dataset = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=True,class_choice='chair')
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
parameters = smoothness_loss_parameters(faces)

network = SVR_TMNet()
network.apply(weights_init)
network.cuda()
if opt.model != '':
    model_dict = network.state_dict()
    pretrained_dict = {k: v for k, v in torch.load(opt.model).items() if (k in model_dict)}
    model_dict.update(pretrained_dict)
    network.load_state_dict(model_dict)
    print(" Previous weight loaded ")
network.estimate2 = network.estimate
print(network)
network.cuda()

lrate = opt.lr
optimizer = optim.Adam([
    {'params':network.decoder2.parameters()},
    {'params':network.estimate2.parameters()}
    ], lr = lrate )


train_l2_loss = AverageValueMeter()
val_l2_loss = AverageValueMeter()
train_CDs_stage2_loss = AverageValueMeter()
val_CDs_stage2_loss = AverageValueMeter()

with open(logname, 'a') as f:
    f.write(str(network) + '\n')

train_l2_curve = []
val_l2_curve = []
train_CDs_stage2_curve = []
val_CDs_stage2_curve = []

for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_CDs_stage2_loss.reset()
    train_l2_loss.reset()
    network.eval()
    network.decoder2.train()
    network.estimate.train()

    if epoch == opt.epoch_decay:
        optimizer = optim.Adam([
            {'params': network.decoder2.parameters()},
            {'params': network.estimate2.parameters()}
        ], lr=lrate/10.0)

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
            pointsRec = network(img, vertices_input, mode='deform1')
            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            error_stage1 = network(img, pointsRec_samples.detach().transpose(1, 2), mode='estimate')
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(img.size(0), faces_cuda.size(0), faces_cuda.size(1))
            faces_cuda_bn = prune(faces_cuda_bn.detach(), error_stage1.detach(), opt.tau, index, opt.pool)

        pointsRec2 = network(img, pointsRec, mode='deform2')
        _, _, _, idx2 = distChamfer(points, pointsRec2)

        pointsRec2_samples, _ = samples_random(faces_cuda_bn.detach(), pointsRec2, opt.num_points)
        dist12_samples, dist22_samples, _, _ = distChamfer(points, pointsRec2_samples)
        choice2 = np.random.choice(points.size(1), opt.num_samples, replace=False)
        error_GT = torch.sqrt(dist22_samples.detach()[:,choice2])
        error = network(img, pointsRec2_samples.detach()[:,choice2].transpose(1, 2), mode='estimate2')

        cds_stage2 = (torch.mean(dist12_samples)) + (torch.mean(dist22_samples))
        l2_loss = calculate_l2_loss(error, error_GT.detach())
        normal_loss = get_normal_loss(pointsRec2, faces_cuda_bn, normals, idx2)
        edge_loss = get_edge_loss(pointsRec2, faces_cuda_bn)
        smoothness_loss = get_smoothness_loss(pointsRec2, parameters, faces_cuda_bn)

        loss_net = cds_stage2 + opt.lambda_normal * normal_loss + opt.lambda_edge * edge_loss + \
                   opt.lambda_smooth * smoothness_loss + l2_loss

        loss_net.backward()
        train_CDs_stage2_loss.update(cds_stage2.item())
        train_l2_loss.update(l2_loss.item())
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
            vis.scatter(X=pointsRec2_samples[0].data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )
        print('[%d: %d/%d] train_cd_loss:  %f  / l2_loss: %f' % (epoch, i, len_dataset / opt.batchSize,
                                                                 cds_stage2.item(),l2_loss.item()))

    train_l2_curve.append(train_l2_loss.avg)
    train_CDs_stage2_curve.append(train_CDs_stage2_loss.avg)

    with torch.no_grad():
        # VALIDATION
        val_CDs_stage2_loss.reset()
        val_l2_loss.reset()
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

            pointsRec = network(img, vertices_input, mode='deform1')  # vertices_sphere 3*2562
            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            error = network(img, pointsRec_samples.detach().transpose(1, 2), mode='estimate')
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(img.size(0), faces_cuda.size(0), faces_cuda.size(1))
            faces_cuda_bn = prune(faces_cuda_bn, error, opt.tau, index, opt.pool)

            pointsRec2 = network(img, pointsRec, mode='deform2')
            _, _, _, idx2 = distChamfer(points, pointsRec2)

            pointsRec2_samples, index = samples_random(faces_cuda_bn, pointsRec2, opt.num_points)
            dist12_samples, dist22_samples, _, _ = distChamfer(points, pointsRec2_samples)
            error_GT = torch.sqrt(dist22_samples.detach())
            error = network(img, pointsRec2_samples.detach().transpose(1, 2), mode='estimate2')

            cds_stage2 = (torch.mean(dist12_samples)) + (torch.mean(dist22_samples))
            l2_loss = calculate_l2_loss(error, error_GT.detach())
            normal_loss = get_normal_loss(pointsRec2, faces_cuda_bn, normals, idx2)
            edge_loss = get_edge_loss(pointsRec2, faces_cuda_bn)
            smoothness_loss = get_smoothness_loss(pointsRec2, parameters, faces_cuda_bn)

            val_CDs_stage2_loss.update(cds_stage2.item())
            val_l2_loss.update(l2_loss.item())
            dataset_test.perCatValueMeter[cat[0]].update(cds_stage2.item())

            if i % 25 == 0:
                vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE TRAIN"))
                vis.scatter(X=points[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsRec2_samples[0].data.cpu(),
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )

            print('[%d: %d/%d] val_cd_loss:  %f , l2_loss: %f' % (epoch, i, len(dataset_test)/opt.batchSize,
                                                                  cds_stage2.item(),l2_loss.item()))

        val_l2_curve.append(val_l2_loss.avg)
        val_CDs_stage2_curve.append(val_CDs_stage2_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_l2_curve)), np.arange(len(val_l2_curve)))),
             Y=np.log(np.column_stack((np.array(train_l2_curve), np.array(val_l2_curve)))),
             win='L2_loss',
             opts=dict(title="L2_loss", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_CDs_stage2_curve)), np.arange(len(val_CDs_stage2_curve)))),
             Y=np.log(np.column_stack((np.array(train_CDs_stage2_curve),np.array(val_CDs_stage2_curve)))),
             win='CDs_stage2',
             opts=dict(title="CDs_stage2", legend=["train", "val"], markersize=2, ), )

    log_table = {
        "train_l2_loss": train_l2_loss.avg,
        "train_cds_stage2": train_CDs_stage2_loss.avg,
        "val_l2_loss": val_l2_loss.avg,
        "val_cds_stage2": val_CDs_stage2_loss.avg,
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
