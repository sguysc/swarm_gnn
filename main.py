# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:34:46 2021

@author: guy
"""

import numpy as np
import os
# import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import PIL.Image
import matplotlib.pyplot as plt
import imageio

# import simulator as sim
import simulator_v3 as sim

# torch has annoying errors
import warnings
warnings.filterwarnings("ignore")

# seed for reproducibility
# np.random.seed(0)
np.random.seed(1)

what_to_load = 1
save_intermediate_steps = False
load_checkpoint_model = True

# load reference image, figure out how many robots needed
max_size = 20
if(what_to_load == 0):
    fname   = 'pic3.png'
    ref_img = PIL.Image.open(fname)
    ref_img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    ref_img = 1.0-np.float32(ref_img)
    number_of_robots = int(ref_img.sum())
    ref_img = th.tensor(ref_img)
    # saving also as the locations that the robot has
    ref_array = th.nonzero(ref_img).float()
    mean_ref_array = th.tensor([th.mean(ref_array[:,0]), th.mean(ref_array[:,1])])
    # breakpoint()
elif(what_to_load == 1):
    max_size = 6
    ref_img = th.zeros(( max_size, max_size ))
    # ref_img[2,1:3] = 1
    # ref_img[1,1:3] = 1
    ref_img[1:5,3] = 1
    number_of_robots = int(ref_img.sum())
    ref_array = th.nonzero(ref_img).float()

step_size = th.tensor(0.1)

# th.autograd.set_detect_anomaly(True)

# class GCNLayer(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super(GCNLayer, self).__init__()
#         self.linear = nn.Linear(in_feats, out_feats)

#     def forward(self, g, feature):
#         # Creating a local scope so that all the stored ndata and edata
#         # (such as the `'h'` ndata below) are automatically popped out
#         # when the scope exits.
#         with g.local_scope():
#             g.ndata['h'] = feature
#             g.update_all(gcn_msg, gcn_reduce)
#             h = g.ndata['h']
#             return self.linear(h)

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats1, h_feats2, num_outputs, dropout=0):
        super(GCN, self).__init__()
        # self.conv1 = GraphConv(in_feats, h_feats1, weight=True, bias=True) # GCNLayer
        # self.conv2 = GraphConv(h_feats1, h_feats2, weight=True, bias=True) # GCNLayer
        # self.conv3 = GraphConv(h_feats2, num_outputs, weight=True, bias=True) # GCNLayer
        self.conv1 = GraphConv(in_feats, h_feats1, weight=True, bias=True) # GCNLayer
        self.conv3 = GraphConv(h_feats1, num_outputs, weight=True, bias=True) # GCNLayer
        if(dropout):
            self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        if(self.dropout):
            h = self.dropout(h)
        h = F.relu(h)
        # h = self.conv2(g, h)
        # if(self.dropout):
            # h = self.dropout(h)
        # h = F.relu(h)
        h = self.conv3(g, h)

        # breakpoint()
        # fire_rate = 0.8
        # update_mask = th.rand(h.shape) <= fire_rate
        # h = h * update_mask.float()
        return h

# def evaluate(model, graph, features, labels, mask):
#     model.eval()
#     with th.no_grad():
#         logits = model(graph, features)
#         logits = logits[mask]
#         labels = labels[mask]
#         _, indices = th.max(logits, dim=1)
#         correct = th.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)
def pairwise_dist(x, y):
    xx, yy, zz = th.mm(x,x.t()), th.mm(y,y.t()), th.mm(x, y.t())
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
    alpha = 0.10 #0.1
    xx, yy, zz = th.mm(x,x.t()), th.mm(y,y.t()), th.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = th.exp(- alpha * (rx.t() + rx - 2*xx))
    L = th.exp(- alpha * (ry.t() + ry - 2*yy))
    P = th.exp(- alpha * (rx.t() + ry - 2*zz))

    B = 10.
    beta = (1./(B*(B-1)))
    gamma = (2./(B*B))

    return beta * (th.sum(K)+th.sum(L)) - gamma * th.sum(P)

def Energy_loss(x,y):
    n_1 = x.shape[0]
    n_2 = y.shape[0]

    a00 = - 1. / (n_1 * n_1)
    a11 = - 1. / (n_2 * n_2)
    a01 = 1. / (n_1 * n_2)

    xy = th.cat((x, y), 0)
    distances = pairwise_dist(xy, xy)
    d_1 = distances[:n_1, :n_1].sum()
    d_2 = distances[-n_2:, -n_2:].sum()
    d_12 = distances[:n_1, -n_2:].sum()

    loss = 2 * a01 * d_12 + a00 * d_1 + a11 * d_2

    return loss

def Distance_Penalty_loss(x):
    # breakpoint()
    # zero distances mean self loops and they are constant per problem,
    # so this would just add a constant
    eps = 0.01
    y = th.where(x < eps, eps*th.ones_like(x), x)

    inv_x = th.reciprocal(y)
    # option 1: the sum of them should be minimized
    # loss = inv_x.sum()
    # breakpoint()
    # option 2: the worst case of them should be minimzed_
    # loss = nn.functional.softmax(inv_x).norm(dim=0)
    # loss = inv_x.max()
    loss = inv_x.sum()
    # option 3
    # loss = -x.min()

    return loss

def train(sim, model, lr=0.001, num_iterations=1000, save_intermediate=False):
    loss_history = np.zeros(num_iterations)

    optimizer = th.optim.Adam(model.parameters(), lr=lr)

    for e in range(num_iterations):
        # get the new graph
        # breakpoint()
        g = sim.G
        features = g.ndata['data']
        # print('features have %d nonzero elements' %features.nonzero().sum())
        # Forward
        logits = model(g, features)
        # first two are dx, dy. the rest is d_data
        d_x    =  step_size*(th.sigmoid( step_size*logits[:,0] ) - .50) #  step_size *
        d_y    =  step_size*(th.sigmoid( step_size*logits[:,1] ) - .50)  # step_size *
        d_data =  step_size*(th.sigmoid( step_size*logits[:,2:] ) - .50) #logits[:,2:]
        # breakpoint()
        # get the new poses with the gradients
        truth_pose_list = sim.get_new_state_list( step_size*(th.sigmoid( step_size*logits[:,0:2] ) - .50))

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        # loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # err = truth_pose - ref_img
        # loss = th.mean(th.mul(err, err))

        # mse = nn.MSELoss()
        # loss1 = mse(truth_pose_list, ref_array) #.mse_loss
        # loss = mse(truth_pose_list, mean_ref_array) #.mse_loss

        # output = loss(truth_pose, ref_img) #.mse_loss
        # sort_arrays(truth_pose_list, ref_array)
        # every 50 iterations, do minimum distance. it helps getting the distribution close
        # to the original reference.
        # if(e % 2 == 1):
        #     loss, vals = NN_loss(truth_pose_list, ref_array)
        # else:
        # loss1 = MMD_loss(truth_pose_list, ref_array)
        # if(e == 49):
        #     breakpoint()
        # create a distance matrix
        dist_sq = th.cdist(truth_pose_list, truth_pose_list, p=2.0)
        # gets the indices of the upper triangle (the distinct distance values)
        ind = th.triu_indices(dist_sq.shape[0],dist_sq.shape[1],1)
        edge_distances = dist_sq[ind[0], ind[1]]
        loss2 = Distance_Penalty_loss(edge_distances) #, my_sim.G.num_edges()-my_sim.G.num_nodes() )
        # loss1 = Energy_loss(truth_pose_list, ref_array)
        kldiv = nn.KLDivLoss()
        loss1 = kldiv(truth_pose_list, ref_array)
        loss = 1.0*loss2 + loss1
        # loss = loss1
        # train_acc = 0.

        # print('loss1=%f, loss2=%f' %(loss1, loss2))
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
        if(e == 0):
            # loss.backward(retain_graph=True)
            loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # now that we've done computing the loss with gradients, update the simulation.
        # the features are not supposed to have gradients
        sim.update(d_x, d_y, d_data)
        if(save_intermediate):
            if 1: #e % 50 == 0:
                sim.plot(txt='step_' + str(i), clear_first=True, color='g')
                sim.fig.savefig('intermediate/step_' + str(e) + '.png')

        if e % 50 == 0:
            print('In epoch {}, loss: {:.3f} (={:.3f}+{:.3f})'.format(
                e, loss, loss1, loss2)) #, train acc: {:.3f} , train_acc

        loss_history[e] = loss.detach().numpy()

    return loss_history


# gcn_msg = fn.copy_src(src='h', out='m')
# gcn_reduce = fn.sum(msg='m', out='h')
plt.close('all')
hdir = 'intermediate'
if(not os.path.exists(hdir)):
    os.mkdir(hdir)
    print("Directory created")
else:
    # clear old data
    for f in os.listdir(hdir):
        os.remove(os.path.join(hdir, f))
    print('removed old files')

# create the simulator
my_sim = sim.Simulator(num_robots=number_of_robots, memory_size=32, \
                       range_len=5., max_x=max_size, max_y=max_size, num_of_nn=3)

# breakpoint()
outputs = 2 + my_sim.memory_size # dx, dy, d_data
model = GCN(my_sim.G.ndata['data'].shape[1], 128, 128, outputs, dropout=0.2)
if(load_checkpoint_model):
    if(os.path.exists('checkpoint_model.pt')):
        checkpoint = th.load('checkpoint_model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.train()
        print('loaded checkpoint model ...')
    else:
        print('checkpoint file not found')

my_sim.plot(ext_list=ref_array, txt='ref.', color='b')
my_sim.plot(txt='init.', init=True, color='r')
# plot for the loss history
fig_h, axs_h = plt.subplots()

num_epochs = 1000
save_intermediate = save_intermediate_steps
for i in range(num_epochs):
    if(i==num_epochs-1):
        # save the last epoch
        save_intermediate = True
    loss_h = train(my_sim, model, lr=1e-4, num_iterations=int(10e1), save_intermediate=save_intermediate)
    # if(i==num_epochs-1):
    my_sim.plot(txt='final_' + str(i), clear_first=True, color='g')
    # my_sim.fig.savefig('iter' + str(i) + '.png')

    axs_h.plot(loss_h, label='iter_'+str(i) )
    # new random locations
    if(i == num_epochs-1):
        break
    # breakpoint()
    my_sim.randomize(keep_init=True)
    my_sim.plot(txt='init.', init=True)
    print('finished epoch #%d' %(i))

plt.xlabel("iterations")
plt.ylabel("loss")
plt.legend(loc='upper left')

# breakpoint()
if(save_intermediate):
    with imageio.get_writer(os.path.join(hdir, 'movie.gif'), mode='I', duration = .4) as writer:
        num_files = len(os.listdir(hdir)) - 1
        for filename in range(num_files):
            image = imageio.imread(os.path.join(hdir, 'step_%d.png'%filename))
            writer.append_data(image)

if(load_checkpoint_model):
    print('saving model checkpoint ...')
    th.save({
            'epoch': num_epochs-1,
            'model_state_dict': model.state_dict(),
            'loss': loss_h[-1]}, 'checkpoint_model.pt')


