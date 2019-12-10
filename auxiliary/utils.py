import os
import random
import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix


# initialize the weighs of the network for Convolutional layers and batchnorm layers
class AverageValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_max(errors, index, fn=5120):
    batch_size = errors.shape[0]
    number = errors.shape[1]
    b = torch.stack([torch.bincount(x,minlength=fn).cumsum(0) for x in index])
    b2 = torch.zeros_like(b)
    b2[:,1:].copy_(b[:,:-1])
    c = torch.LongTensor(range(number)).expand([batch_size,number])
    index2 = c-torch.gather(b2, 1, index)
    max_errors = []
    for i in range(batch_size):
        row = index[i]
        col = index2[i]
        data = errors[i]
        coo = coo_matrix((data, (row, col)), shape=(fn, int(b[i].max())))
        max_errors.append(torch.from_numpy(coo.max(axis=1).toarray()))
    max_errors = torch.stack(max_errors)
    return max_errors


def get_edges(faces):
    edge = []
    for i, j in enumerate(faces):
        edge.append(j[:2])
        edge.append(j[1:])
        edge.append(j[[0, 2]])
    edge = np.array(edge)
    edge_im = edge[:, 0] * edge[:, 1] + (edge[:, 0] + edge[:, 1]) * 1j
    unique = np.unique(edge_im, return_index=True)[1]
    edge_unique = edge[unique]
    edge_cuda = (torch.from_numpy(edge_unique).type(torch.cuda.LongTensor)).detach()
    return edge_cuda


def get_selected_pair(faces):
    vertices_number = faces.max().item() + 1

    triangles_new = faces.cpu().data.numpy()
    triangles_new = triangles_new[triangles_new.sum(1).nonzero()]

    edges = np.concatenate((triangles_new[:, :2], triangles_new[:, [0, 2]], triangles_new[:, 1:]), 0)
    edges.sort(1)
    edges_unrolled = edges[:, 0] * vertices_number + edges[:, 1]  # edges extended to 1d

    all_index = np.arange(0, edges_unrolled.shape[0])
    unique_index = np.unique(edges_unrolled, return_index=True)[1]
    repeated_index = np.array(list(set(all_index).difference(set(unique_index))))
    unique_value = edges_unrolled[unique_index]
    repeated_value = edges_unrolled[repeated_index]

    boundary_value = np.array(list(set(unique_value).difference(set(repeated_value))))
    boundary_edge1 = np.array(np.floor(boundary_value / vertices_number))
    boundary_edge2 = np.array(boundary_value % vertices_number)
    boundary_edge = np.stack((boundary_edge1, boundary_edge2), 1)
    boundary_point = np.unique(np.concatenate((boundary_edge1, boundary_edge2), 0))
    boundary_edge_inverse = boundary_edge[:, [1, 0]]
    boundary_edge_all = np.concatenate((boundary_edge, boundary_edge_inverse), 0)
    boundary_edge_all = boundary_edge_all[np.argsort(boundary_edge_all[:, 0])]

    selected_point = np.where(boundary_edge_all[:, 0] == np.concatenate((boundary_edge_all[1:, 0],
                                                                         boundary_edge_all[:1, 0]), 0))
    selected_pair = np.concatenate((boundary_edge_all[selected_point[0]],
                                    boundary_edge_all[selected_point[0] + 1][:, 1:]), 1)
    selected_pair = torch.from_numpy(selected_pair).type(torch.cuda.LongTensor)
    boundary_point = torch.from_numpy(boundary_point).type(torch.cuda.LongTensor)

    return selected_pair, boundary_point, boundary_edge


def samples_random(faces_cuda, pointsRec, sampled_number):

    if len(faces_cuda.size())==2:
        faces_points = pointsRec.index_select(1, faces_cuda.view(-1)).contiguous().\
            view(pointsRec.size()[0], faces_cuda.size()[0], faces_cuda.size()[1], pointsRec.size()[2])
    elif len(faces_cuda.size())==3:
        faces_points = pointsRec.index_select(1, faces_cuda.view(-1)).contiguous().\
            view(pointsRec.size()[0] ** 2, faces_cuda.size()[1], faces_cuda.size()[2], pointsRec.size()[2])
        index = (torch.arange(0, pointsRec.size()[0]) * (pointsRec.size()[0] + 1)).\
            type(torch.cuda.LongTensor)
        faces_points = faces_points.index_select(0,index)
    else:
        faces_points = None

    faces_points_np = faces_points.cpu().data.numpy()
    a = faces_points_np[:, :, 0]
    b = faces_points_np[:, :, 1]
    c = faces_points_np[:, :, 2]

    cross = np.cross(b - a, c - a)
    area = np.sqrt(cross[:, :, 0] ** 2 + cross[:, :, 1] ** 2 + cross[:, :, 2] ** 2)
    area_sum = np.sum(area, axis=1)
    area_cum = np.cumsum(area, axis=1)
    faces_pick = area_sum[:, np.newaxis] * np.random.random(sampled_number)[np.newaxis, :]  # 32*7500

    faces_index = []
    for i in range(faces_pick.shape[0]):
        faces_index.append(np.searchsorted(area_cum[i], faces_pick[i]))

    faces_index = np.array(faces_index)  # 32*7500 the index of faces to be sampled
    faces_index = np.clip(faces_index,0,area_cum.shape[1]-1)
    faces_index_tensor = (torch.from_numpy(faces_index).cuda()).type(torch.cuda.LongTensor)
    faces_index_tensor_sort = faces_index_tensor.sort(1)[0]

    tri_origins = faces_points[:, :, 0].clone()
    tri_vectors = faces_points[:, :, 1:].clone()
    tri_vectors = tri_vectors - tri_origins.unsqueeze(2).expand_as(tri_vectors)

    tri_origins = tri_origins.index_select(1, faces_index_tensor_sort.view(-1)).view(
        tri_origins.size()[0] * faces_index_tensor_sort.size()[0],
        faces_index_tensor_sort.size()[1], tri_origins.size()[2])
    tri_vectors = tri_vectors.index_select(1, faces_index_tensor_sort.view(-1)).view(
        tri_vectors.size()[0] * faces_index_tensor_sort.size()[0],
        faces_index_tensor_sort.size()[1], tri_vectors.size()[2], tri_vectors.size()[3])

    diag_index = (torch.arange(0, pointsRec.size()[0]).cuda())
    diag_index=diag_index.type(torch.cuda.LongTensor)
    diag_index = (1+faces_index_tensor_sort.size(0)) * diag_index

    tri_origins = tri_origins.index_select(0, diag_index)
    tri_vectors = tri_vectors.index_select(0, diag_index)

    random_lenghts = ((torch.randn(pointsRec.size()[0], tri_origins.size()[1], 2, 1).uniform_(0, 1)).cuda())
    random_test = random_lenghts.sum(2).squeeze(2) > 1.0
    random_test_minus = random_test.type(torch.cuda.FloatTensor).unsqueeze(2).unsqueeze(3).repeat(1, 1, 2, 1)
    random_lenghts = torch.abs(random_lenghts - random_test_minus)
    random_lenghts = random_lenghts.repeat(1,1,1,3)

    sample_vector = (tri_vectors * random_lenghts).sum(2)
    samples = sample_vector + tri_origins

    return samples, faces_index_tensor_sort


def get_boundary_points_bn(faces_cuda_bn, pointsRec_refined):
    selected_pair_all = []
    selected_pair_all_len = []
    boundary_points_all = []
    boundary_points_all_len = []

    for bn in torch.arange(0, faces_cuda_bn.shape[0]):
        faces_each = faces_cuda_bn[bn]
        selected_pair, boundary_point, _ = get_selected_pair(faces_each)
        selected_pair_all.append(selected_pair)
        selected_pair_all_len.append(len(selected_pair))
        boundary_points_all.append(boundary_point)
        boundary_points_all_len.append(len(boundary_point))
    max_len = np.array(selected_pair_all_len).max()
    max_len2 = np.array(boundary_points_all_len).max()
    for bn in torch.arange(0, faces_cuda_bn.shape[0]):
        if len(selected_pair_all[bn]) < max_len:
            lencat = max_len - len(selected_pair_all[bn])
            tensorcat = torch.zeros(lencat, 3).type_as(selected_pair_all[bn])
            selected_pair_all[bn] = torch.cat((selected_pair_all[bn], tensorcat), 0)
        if len(boundary_points_all[bn]) < max_len2:
            lencat = max_len2 - len(boundary_points_all[bn])
            if len(boundary_points_all[bn]) > 0:
                tensorcat = torch.Tensor(lencat).fill_(boundary_points_all[bn][0]).type_as(boundary_points_all[bn])
            else:
                tensorcat = torch.zeros(lencat).type_as(boundary_points_all[bn])
            boundary_points_all[bn] = torch.cat((boundary_points_all[bn], tensorcat), 0)

    selected_pair_all = torch.stack(selected_pair_all, 0)
    selected_pair_all_len = np.array(selected_pair_all_len)
    indices = (torch.arange(0, faces_cuda_bn.size(0)) * (1 + faces_cuda_bn.size(0))).type(torch.cuda.LongTensor)
    pointsRec_refined_boundary = pointsRec_refined.index_select(1, selected_pair_all.view(-1)). \
        view(pointsRec_refined.shape[0] * selected_pair_all.shape[0], selected_pair_all.shape[1],
             selected_pair_all.shape[2], pointsRec_refined.shape[2])
    pointsRec_refined_boundary = pointsRec_refined_boundary.index_select(0, indices)

    return pointsRec_refined_boundary, selected_pair_all, selected_pair_all_len


def prune(faces_cuda_bn, error, tau, index, pool='max', faces_number=5120):
    error = torch.pow(error, 2)
    if not pool == 'sum':
        tau = tau / 10.0
    ones = (torch.ones(1).cuda()).expand_as(error).type(torch.cuda.FloatTensor)
    zeros = (torch.Tensor(error.size(0) * faces_cuda_bn.size(1)).fill_(0)).cuda()
    index_1d = (index + (torch.arange(0, error.size(0)).unsqueeze(1).
                         expand_as(index) * faces_cuda_bn.size(1)).type(torch.cuda.LongTensor)).view(-1)
    face_error = zeros.index_add_(0, index_1d, error.view(-1)).view(error.size(0), faces_cuda_bn.size(1))
    face_count = zeros.index_add(0, index_1d, ones.view(-1)).view(error.size(0), faces_cuda_bn.size(1))
    faces_cuda_bn = faces_cuda_bn.clone()

    if pool == 'mean':
        face_error = face_error / (face_count + 1e-12)
    elif pool == 'max':
        face_error = get_max(error.cpu(), index.cpu(), faces_number)
        face_error = face_error.squeeze(2).cuda()
    elif pool == 'sum':
        face_error = face_error
    faces_cuda_bn[face_error > tau] = 0

    faces_cuda_set = []
    for k in torch.arange(0, error.size(0)):
        faces_cuda = faces_cuda_bn[k]
        _, _, boundary_edge = get_selected_pair(faces_cuda)
        boundary_edge_point = boundary_edge.astype(np.int64).reshape(-1)
        counts = pd.value_counts(boundary_edge_point)
        toremove_point = torch.from_numpy(np.array(counts[counts > 2].index)).cuda()
        faces_cuda_expand = faces_cuda.unsqueeze(2).expand(faces_cuda.shape[0], faces_cuda.shape[1],
                                                           toremove_point.shape[0])
        toremove_point_expand = toremove_point.unsqueeze(0).unsqueeze(0).\
            expand(faces_cuda.shape[0],faces_cuda.shape[1],toremove_point.shape[0])
        toremove_index = ((toremove_point_expand == faces_cuda_expand).sum(2).sum(1)) != 0
        faces_cuda[toremove_index] = 0
        triangles = faces_cuda.cpu().data.numpy()

        v = pd.value_counts(triangles.reshape(-1))
        v = v[v == 1].index
        for vi in v:
            if np.argwhere(triangles == vi).shape[0] == 0:
                continue
            triangles[np.argwhere(triangles == vi)[0][0]] = 0

        faces_cuda_set.append(torch.from_numpy(triangles).cuda().unsqueeze(0))
    faces_cuda_bn = torch.cat(faces_cuda_set, 0)

    return faces_cuda_bn
