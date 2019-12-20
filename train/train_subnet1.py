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
parser.add_argument('--nepoch', type=int, default=420, help='number of epochs to train for')
parser.add_argument('--epoch_decay', type=int, default=300, help='epoch to decay lr')
parser.add_argument('--epoch_decay2', type=int, default=400, help='epoch to decay lr for the second time')
parser.add_argument('--model', type=str, default='./log/pretrain/network.pth', help='model path from the pretrained model')
parser.add_argument('--num_points', type=int, default=10000, help='number of points for GT point cloud')
parser.add_argument('--num_vertices', type=int, default=2562, help='number of vertices of the initial sphere')
parser.add_argument('--num_samples',type=int,default=2500, help='number of samples for error estimation')
parser.add_argument('--env', type=str, default="SVR_subnet1", help='visdom env')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--tau', type=float, default=0.1, help='threshold to prune the faces')
parser.add_argument('--lambda_edge', type=float, default=0.05, help='weight of edge loss')
parser.add_argument('--lambda_smooth', type=float, default=5e-7, help='weight of smooth loss')
parser.add_argument('--lambda_normal', type=float, default=1e-3, help='weight of normal loss')
parser.add_argument('--pool', type=str, default='max', help='max or mean or sum')
parser.add_argument('--manualSeed', type=int, default=6185)
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

dataset = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=True, class_choice='chair')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataset_test = ShapeNet(npoints=opt.num_points, SVR=True, normal=True, train=False, class_choice='chair')
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
print(network)
network.cuda()

lrate = opt.lr
optimizer = optim.Adam([
    {'params': network.encoder.parameters()},
    {'params': network.estimate.parameters()},
    {'params': network.decoder.parameters()}
], lr=lrate)

train_CD_loss = AverageValueMeter()
val_CD_loss = AverageValueMeter()
train_l2_loss = AverageValueMeter()
val_l2_loss = AverageValueMeter()
train_CDs_loss = AverageValueMeter()
val_CDs_loss = AverageValueMeter()


with open(logname, 'a') as f:
    f.write(str(network) + '\n')

train_CD_curve = []
val_CD_curve = []
train_l2_curve = []
val_l2_curve = []
train_CDs_curve = []
val_CDs_curve = []


for epoch in range(opt.nepoch):
    # TRAIN MODE
    train_CD_loss.reset()
    train_CDs_loss.reset()
    train_l2_loss.reset()
    network.train()

    if epoch == opt.epoch_decay:
        optimizer = optim.Adam([
            {'params': network.encoder.parameters()},
            {'params': network.estimate.parameters()},
            {'params': network.decoder.parameters()}
        ], lr=lrate / 10.0)

    if epoch == opt.epoch_decay2:
        optimizer = optim.Adam([
            {'params': network.encoder.parameters()},
            {'params': network.estimate.parameters()},
            {'params': network.decoder.parameters()}
        ], lr=lrate / 100.0)

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
        pointsRec = network(img, vertices_input, mode='deform1')  # vertices_sphere 3*2562
        dist1, dist2, _, idx2 = distChamfer(points_choice, pointsRec)

        pointsRec_samples, _ = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
        dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples.detach())
        choice2 = np.random.choice(points.size(1), opt.num_samples, replace=False)
        error_GT = torch.sqrt(dist2_samples.detach()[:,choice2])
        error = network(img, pointsRec_samples.detach()[:,choice2].transpose(1, 2), mode='estimate')

        CD_loss = torch.mean(dist1) + torch.mean(dist2)
        CDs_loss = torch.mean(dist1_samples) + torch.mean(dist2_samples)
        l2_loss = calculate_l2_loss(error, error_GT.detach())
        edge_loss = get_edge_loss_stage1(pointsRec, edge_cuda.detach())
        smoothness_loss = get_smoothness_loss_stage1(pointsRec, parameters)
        faces_cuda_bn = faces_cuda.unsqueeze(0).expand(pointsRec.size(0), faces_cuda.size(0),faces_cuda.size(1))
        normal_loss = get_normal_loss(pointsRec, faces_cuda_bn, normals, idx2)

        loss_net = CD_loss + l2_loss + opt.lambda_edge * edge_loss + opt.lambda_smooth * smoothness_loss \
                   + opt.lambda_normal * normal_loss

        loss_net.backward()
        train_CD_loss.update(CD_loss.item())
        train_CDs_loss.update(CDs_loss.item())
        train_l2_loss.update(l2_loss.item())
        optimizer.step()

        # VIZUALIZE
        if i % 50 <= 0:
            vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE TRAIN', opts=dict(title="INPUT IMAGE TRAIN"))
            vis.scatter(X=points[0].data.cpu(),
                        win='TRAIN_INPUT',
                        opts=dict(
                            title="TRAIN_INPUT",
                            markersize=2,
                        ),
                        )
            vis.scatter(X=pointsRec[0].data.cpu(),
                        win='TRAIN_INPUT_RECONSTRUCTED',
                        opts=dict(
                            title="TRAIN_INPUT_RECONSTRUCTED",
                            markersize=2,
                        ),
                        )
        print('[%d: %d/%d] train_cd_loss:  %f , l2_loss: %f' % (epoch, i, len_dataset / opt.batchSize,
                                                                     CD_loss.item(),l2_loss.item()))

    train_CD_curve.append(train_CD_loss.avg)
    train_CDs_curve.append(train_CDs_loss.avg)
    train_l2_curve.append(train_l2_loss.avg)

    with torch.no_grad():
        val_CD_loss.reset()
        val_CDs_loss.reset()
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
            pointsRec = network(img, vertices_input, mode='deform1')  # points_sphere 3*2562
            dist1, dist2, _, idx2 = distChamfer(points_choice, pointsRec)

            pointsRec_samples, index = samples_random(faces_cuda, pointsRec.detach(), opt.num_points)
            dist1_samples, dist2_samples, _, _ = distChamfer(points, pointsRec_samples)
            error_GT = torch.sqrt(dist2_samples)
            error = network(img, pointsRec_samples.detach().transpose(1, 2), mode='estimate')

            CD_loss = torch.mean(dist1) + torch.mean(dist2)
            edge_loss = get_edge_loss_stage1(pointsRec, edge_cuda.detach())
            smoothness_loss = get_smoothness_loss_stage1(pointsRec, parameters)
            l2_loss = calculate_l2_loss(error, error_GT.detach())
            CDs_loss = (torch.mean(dist1_samples)) + (torch.mean(dist2_samples))
            faces_cuda_bn = faces_cuda.unsqueeze(0).expand(error.size(0), faces_cuda.size(0), faces_cuda.size(1))
            normal_loss = get_normal_loss(pointsRec, faces_cuda_bn, normals, idx2)

            val_CD_loss.update(CD_loss.item())
            dataset_test.perCatValueMeter[cat[0]].update(CDs_loss.item())
            val_l2_loss.update(l2_loss.item())
            val_CDs_loss.update(CDs_loss.item())

            if i % 25 == 0:
                vis.image(img[0].data.cpu().contiguous(), win='INPUT IMAGE VAL', opts=dict(title="INPUT IMAGE TRAIN"))
                vis.scatter(X=points[0].data.cpu(),
                            win='VAL_INPUT',
                            opts=dict(
                                title="VAL_INPUT",
                                markersize=2,
                            ),
                            )
                vis.scatter(X=pointsRec[0].data.cpu(),
                            win='VAL_INPUT_RECONSTRUCTED',
                            opts=dict(
                                title="VAL_INPUT_RECONSTRUCTED",
                                markersize=2,
                            ),
                            )

            print('[%d: %d/%d] val_cd_loss:  %f , l2_loss: %f' % (epoch, i, len(dataset_test) / opt.batchSize,
                                                                  CD_loss.item(), l2_loss.item()))

        val_CD_curve.append(val_CD_loss.avg)
        val_l2_curve.append(val_l2_loss.avg)
        val_CDs_curve.append(val_CDs_loss.avg)

    vis.line(X=np.column_stack((np.arange(len(train_CD_curve)), np.arange(len(val_CD_curve)))),
             Y=np.log(np.column_stack((np.array(train_CD_curve), np.array(val_CD_curve)))),
             win='CD_vertices',
             opts=dict(title="CD_vertices", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_l2_curve)), np.arange(len(val_l2_curve)))),
             Y=np.log(np.column_stack((np.array(train_l2_curve), np.array(val_l2_curve)))),
             win='L2_loss',
             opts=dict(title="L2_loss", legend=["train", "val"], markersize=2, ), )

    vis.line(X=np.column_stack((np.arange(len(train_CDs_curve)),np.arange(len(val_CDs_curve)))),
             Y=np.log(np.column_stack((np.array(train_CDs_curve),np.array(val_CDs_curve)))),
             win='CD_samples',
             opts=dict(title="CD_samples", legend=["train", "val"], markersize=2, ), )

    log_table = {
        "train_CD_loss": train_CD_loss.avg,
        "val_CD_loss": val_CD_loss.avg,
        "train_l2_loss": train_l2_loss.avg,
        "val_l2_loss": val_l2_loss.avg,
        "val_CDs_loss": val_CDs_loss.avg,
        "epoch": epoch,
        "lr": lrate,
    }
    print(log_table)
    for item in dataset_test.cat:
        print(item, dataset_test.perCatValueMeter[item].avg)
        log_table.update({item: dataset_test.perCatValueMeter[item].avg})
    with open(logname, 'a') as f:
        f.write('json_stats: ' + json.dumps(log_table) + '\n')
    torch.save(network.state_dict(), '%s/network.pth' % (dir_name))
