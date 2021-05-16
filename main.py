# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:34:46 2021

@author: guy
"""

import numpy as np
# import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import PIL.Image

# import simulator as sim
import simulator_v3 as sim


# seed for reproducibility
np.random.seed(0)

# load reference image, figure out how many robots needed
max_size = 40
fname   = 'pic1.png'
ref_img = PIL.Image.open(fname)
ref_img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
ref_img = 1.0-np.float32(ref_img)
number_of_robots = int(ref_img.sum())
ref_img = torch.tensor(ref_img)
# saving also as the locations that the robot has
ref_array = torch.nonzero(ref_img).float()

# create the simulator
my_sim = sim.Simulator(num_robots=number_of_robots, memory_size=64, \
                       range_len=3., max_x=max_size, max_y=max_size)
step_size = torch.tensor(0.1)

torch.autograd.set_detect_anomaly(True)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_outputs):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_outputs)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# def evaluate(model, graph, features, labels, mask):
#     model.eval()
#     with torch.no_grad():
#         logits = model(graph, features)
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)
def pairwise_dist(x, y):
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    P = (rx.t() + ry - 2*zz)
    return P

def NN_loss(x, y, dim=0):
    # breakpoint()
    dist = pairwise_dist(x.float(), y.float())
    values, indices = dist.min(dim=dim)
    return values.mean(), values

#https://stats.stackexchange.com/questions/276497/maximum-mean-discrepancy-distance-distribution
def MMD_loss(x, y):
    alpha = 0.1
    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- alpha * (rx.t() + rx - 2*xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2*yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2*zz))

    B = 10.
    beta = (1./(B*(B-1)))
    gamma = (2./(B*B)) 

    return beta * (torch.sum(K)+torch.sum(L)) - gamma * torch.sum(P)
    
def train(g, model):
    # breakpoint()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # best_val_acc = 0
    # best_test_acc = 0

    features = g.ndata['data']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    for e in range(100):
        # breakpoint()
        # Forward
        logits = model(g, features)
        # first two are dx, dy. the rest is d_data
        d_x    = torch.sign( logits[:,0] )
        d_y    = torch.sign( logits[:,1] )
        d_data = logits[:,2:]
        # breakpoint()
        my_sim.update(d_x, d_y, d_data)
        # truth_pose = my_sim.get_state_map()
        truth_pose_list = my_sim.get_state_list()
        # # Compute prediction
        # pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # err = truth_pose - ref_img
        # loss = torch.mean(torch.mul(err, err))
        # breakpoint()
        # loss = nn.MSELoss()
        # output = loss(truth_pose_list, ref_array) #.mse_loss
        # output = loss(truth_pose, ref_img) #.mse_loss
        # sort_arrays(truth_pose_list, ref_array)
        # loss, vals = NN_loss(truth_pose_list, ref_array)
        loss = MMD_loss(truth_pose_list, ref_array)
        train_acc = 0.
        
        print(loss)
        # print(output)
        
        # Compute accuracy on training/validation/test
        # train_acc = (truth_pose == ref_img).float().mean()
        # train_acc = vals.float().mean()
        # val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        # test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        # if best_val_acc < val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, train acc: {:.3f}'.format(
                e, loss, train_acc))
            

# Create the model with given dimensions
outputs = 2 + my_sim.memory_size # dx, dy, d_data
model = GCN(my_sim.G.ndata['data'].shape[1], 16, outputs)
train(my_sim.G, model)