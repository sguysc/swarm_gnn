# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:53:31 2021

@author: guy
"""

import numpy as np
import dgl
import torch as th
import dgl.function as fn

import matplotlib.pyplot as plt

# helper class, this is essentialy the simulator
class Simulator(object):
    def __init__(self, num_robots=10, memory_size=10, range_len=2., max_x=10, max_y=10, num_of_nn=5, seed=None):
        # id, x, y, (xrel,yrel)*nn, features
        self.robots = th.zeros(num_robots, 1 + 2 + num_of_nn*2 + memory_size, requires_grad=False)
        self.n = num_robots
        self.range_len = range_len
        self.memory_size = memory_size
        self.fsize = (max_x, max_y)
        if(num_of_nn >= num_robots):
            self.num_of_nn = num_robots-1
        else:
            self.num_of_nn = num_of_nn
        # self.map = th.zeros(max_x, max_y, requires_grad=True)

        self.fig, self.axs = plt.subplots()
        self.last_line = None
        self.ref_line = None
        self.init_line = None

        # breakpoint()

        self.randomize(first_time=True, keep_init=False, seed=seed)
        # self.create_graph()

    def randomize(self, first_time=False, keep_init=False, seed=None):
        # print('start randomizing')
        if(keep_init == False):
            init_locations = []
            for rid in range(self.n):
                if(seed is not None):
                    x = seed[rid, 0]
                    y = seed[rid, 1]
                else:
                    # sample an initial location for this robot. the -0.1 is so we don't have a robot on the max index
                    [x, y] = np.random.randint(0, self.fsize[0], 2)
                    while([x,y] in init_locations):
                        [x, y] = np.random.randint(0, self.fsize[0], 2)
                init_locations.append([x, y])

                self.robots[rid,0] = rid
                self.robots[rid,1] = x
                self.robots[rid,2] = y
                self.robots[rid,3:3+self.num_of_nn*2] = th.zeros(self.num_of_nn*2, requires_grad=False) # the relative position of 5 closest robots
                self.robots[rid,3+self.num_of_nn*2:] = th.ones(self.memory_size, requires_grad=False)
                # self.map[x,y] = 1

            # breakpoint()
            self.populate_rel_positions()

            self.saved_robots = th.clone(self.robots)
            # print(self.saved_robots[:, 1:3])
        else:
            # load it again
            self.robots = th.clone(self.saved_robots)
            self.robots[:, 1:3] += 0.01 * th.randn((self.saved_robots.shape[0], 2))
            self.populate_rel_positions()

        # print('done randomizing')
        self.map_list = self.robots[:,1:3]
        self.create_graph(first_time=first_time)

    def populate_rel_positions(self, data=None):
        if(data):
            X = data
        else:
            X = self.robots[:, 1:3]
        dist_sq = th.cdist(X, X, p=2.0)
        knn = dist_sq.topk(self.num_of_nn+1, largest=False)
        j=3 # because of id, x, y
        for nn in range(self.num_of_nn):
            # the first index is itself, so that distance is zero and it doesn't count
            self.robots[:, j ]   = X[knn.indices[:, nn+1], 0] - self.robots[:, 1 ] # dx
            self.robots[:, j+1 ] = X[knn.indices[:, nn+1], 1] - self.robots[:, 2 ] # dy
            j += 2

    def create_graph(self, first_time=True):
        # get all current locations.
        # option 1:
        # X = self.robots[:, 1:3].detach().numpy()
        # # create a distance matrix
        # dist_sq = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = -1))
        # # clear all the connections from robots that are too far away
        # dist_sq[dist_sq > self.range_len] = 0.
        # # create the graph connections
        # self.Gnx = nx.from_numpy_matrix(dist_sq)
        # self.G = dgl.from_networkx(self.Gnx)
        # option 2:
        # breakpoint()
        X = self.robots[:, 1:3]
        # create a distance matrix
        dist_sq = th.cdist(X, X, p=2.0)
        # adjacency matrix
        adj_matrix = th.where(dist_sq <= self.range_len, th.ones_like(dist_sq), th.zeros_like(dist_sq)) # th.zeros_like(dist_sq))
        # adj_matrix = th.where(dist_sq <= self.range_len, dist_sq, -1.0*th.ones_like(dist_sq)) # th.zeros_like(dist_sq))
        # adj_matrix = th.where(adj_matrix >= 0., th.ones_like(dist_sq), th.zeros_like(dist_sq))
        # list of nodes connected
        conn_nodes = th.nonzero(adj_matrix>0.5)
        if(first_time):
            # create a graph, src nodes -> dest nodes. they are bi-directional
            self.G = dgl.graph((conn_nodes[:,0], conn_nodes[:,1]))
            # breakpoint()
            # GUY TODO: check this, it is meant to allow for zero degree-in nodes (in case every
            # other node is too far)
            self.G = dgl.add_self_loop(self.G)

            if(th.cuda.is_available()):
                device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
                if("cpu" not in device ):
                    self.G = self.G.to(device)
        else:
            # use the same graph, remove all nodes and re-create them. don't know if that's the right thing to do
            # edge_ids = th.arange(0, self.G.num_edges())
            # self.G.remove_edges(edge_ids)
            # # add the new ones
            # self.G = dgl.add_edges(self.G, conn_nodes[:,0], conn_nodes[:,1])
            # self.G = dgl.add_self_loop(self.G)
            self.G = dgl.graph((conn_nodes[:,0], conn_nodes[:,1]))
            self.G = dgl.add_self_loop(self.G)

        # add the features to the nodes and edges
        # self.G.ndata['x']     = self.robots[:, 1]
        # self.G.ndata['y']     = self.robots[:, 2]
        # data is essentially all we throw to the neural network. we don't want
        # the nn to learn things using the position / id
        # if(first_time):
        try:
            self.G.ndata['h']  = self.robots[:, 3:].detach()
        except:
            print('all robots are too far appart, the graph is empty')
            breakpoint()

        # else:
        #     data = [ [r.data.tolist() ] for r in self.robots]
        #     data = np.array(data).squeeze()
        #     self.G.ndata['data']  = th.tensor(data)

        # self.G.ndata['id']    = self.robots[:, 0]
        #th.linspace(0, self.G.num_nodes()-1, self.G.num_nodes(), dtype=th.int32).reshape(self.G.num_nodes(),-1)
        # GUY TODO: get the true ranges. for now, we treat all as equal
        nodes = self.G.all_edges()
        tmp  = th.tensor([dist_sq[u,v] for u,v in zip(nodes[0], nodes[1])])
        # take care of what's about to be NaNs
        tmp = th.where(tmp < 0.01, 0.01*th.ones_like(tmp), tmp)
        # in this way, the closer they are, the more weight they'll have
        self.G.edata['w'] = th.reciprocal(tmp)

        # collect features from source nodes and aggregate them in destination nodes
        # g.ndata['h'] stores the input node features
        # self.G.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'data'))
        # g.edata['w'] stores the edge weights
        self.G.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'data'))


    def update(self, d_x, d_y, d_data):
        # option 1:
        # GUY TODO: add some "physics" here. meaning, if they are too close, maybe
        # just get them to be tangent.
        # option 2:
        # breakpoint()
        # the feature inputs must not have gradients (because we don't want to learn them)
        # but the features are self.robots. so basically we update in temp variable and
        # then get them back;
        tmp = self.robots.detach()
        tmp[:, 1]   = th.clamp(tmp[:, 1] + d_x , min=0., max=self.fsize[0])
        tmp[:, 2]   = th.clamp(tmp[:, 2] + d_y , min=0., max=self.fsize[1])
        # not updating the rel positions, only the features
        tmp[:, 3+self.num_of_nn*2:] += d_data
        self.robots = tmp.detach()
        self.populate_rel_positions()

        self.create_graph(first_time=False)

    # def get_state_map(self):
    #     # self.map *= 0.
    #     # breakpoint()
    #     zero_mat = th.zeros(self.map.shape)
    #     self.map = th.matmul(self.map, zero_mat)
    #     one = th.ones(1, dtype=th.int)
    #     for r in range(self.n):
    #         # if it is in a cell of 1x1, mark it
    #         self.map[self.robots[r, 1].int(), self.robots[r, 2].int()] = one #d_x[r]/d_x[r]
    #     return self.map

    def get_new_state_list(self, d_xy):
        # the sigmoid is just to make it bounded by -0.5 to +0.5 without losing gradients (like in sign)
        self.map_list = self.robots[:,1:3] + d_xy
        return self.map_list

    def plot(self, ext_list=None, txt='', clear_first=False, init=False, color='g'):
        if(ext_list is None):
            X = self.robots[:, 1:3]
        else:
            X = ext_list

        # self.axs.clear()
        if(init):
            if(self.init_line):
                self.init_line.remove()
            self.init_line = self.axs.scatter(X[:,1], X[:,0], marker='s', alpha=0.5, label=txt, c=color)
        elif(clear_first):
            if(self.last_line):
                # self.init_line.remove()
                self.last_line.remove()
            self.last_line = self.axs.scatter(X[:,1], X[:,0], marker='s', alpha=0.5, label=txt, c=color)
        else:
            self.ref_line = self.axs.scatter(X[:,1], X[:,0], marker='s', alpha=0.5, label=txt, c=color)

        plt.xlabel("X")
        plt.ylabel("Y")
        self.axs.set_ylim(self.fsize[1], 0)  #
        self.axs.set_xlim(0, self.fsize[0])
        self.axs.legend(loc='upper left')
        plt.show()


if __name__=='__main__':
    w = Simulator(num_robots=5, memory_size=10, range_len=5.)

