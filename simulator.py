# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:53:31 2021

@author: guy
"""

import numpy as np
import networkx as nx
import dgl
import torch as th


# helper class for the robot (aka graph node) attributes
class Robot(object):
    def __init__(self, x, y, rid, memory_size=10):
        self.x = x
        self.y = y
        self.id = rid
        self.data = np.zeros(memory_size)
        
    def move(self, vx, vy, d_data):
        self.x += vx
        self.y += vy
        self.data += d_data
        
# helper class, this is essentialy the simulator
class Simulator(object):
    def __init__(self, num_robots=10, memory_size=10, range_len=2., max_x=10., max_y=10.):

        self.robots = []
        self.n = num_robots
        self.range_len = range_len
        self.memory_size = memory_size
        self.fsize = (max_x, max_y)
        for rid in range(num_robots):
            # sample an initial location for this robot. the -0.1 is so we don't have a robot on the max index
            x = np.around((max_x-0.1)*np.random.rand(), decimals=1)
            y = np.around((max_y-0.1)*np.random.rand(), decimals=1)
            self.robots.append(Robot(x, y, rid, memory_size=memory_size))
        
        self.create_graph()
            
    def create_graph(self, first_time=True):
        # get all current locations.
        X = [ [r.x, r.y] for r in self.robots]
        X = np.array(X)
        
        # create a distance matrix
        dist_sq = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = -1))
        # clear all the connections from robots that are too far away
        dist_sq[dist_sq > self.range_len] = 0.
        # create the graph connections
        self.Gnx = nx.from_numpy_matrix(dist_sq)
        
        # for i in range( self.n ):
        #     self.Gnx.nodes[i]['x']    = self.robots[i].x
        #     self.Gnx.nodes[i]['y']    = self.robots[i].y
        #     self.Gnx.nodes[i]['data'] = self.robots[i].data
        #     self.Gnx.nodes[i]['id']   = i
            
        self.G = dgl.from_networkx(self.Gnx)
        # add the features to the nodes and edges
        
        self.G.ndata['x']     = th.tensor(X[:,0]) #th.tensor(X[:,0:1]) #th.zeros(self.G.num_nodes(), 1)
        self.G.ndata['y']     = th.tensor(X[:,1]) #th.tensor(X[:,1: ]) #th.zeros(self.G.num_nodes(), 1)
        # data is essentially all we throw to the neural network. we don't want
        # the nn to learn things using the position / id
        if(first_time):
            self.G.ndata['data']  = th.zeros(self.G.num_nodes(), self.memory_size, requires_grad=True)
        else:
            data = [ [r.data.tolist() ] for r in self.robots]
            data = np.array(data).squeeze()
            self.G.ndata['data']  = th.tensor(data)
        self.G.ndata['id']    = th.linspace(0, self.G.num_nodes()-1, self.G.num_nodes(), dtype=th.int32)
        #th.linspace(0, self.G.num_nodes()-1, self.G.num_nodes(), dtype=th.int32).reshape(self.G.num_nodes(),-1)
        # GUY TODO: get the true ranges. for now, we treat all as equal
        self.G.edata['range'] = th.ones(self.G.num_edges(), 1)
        # GUY TODO: check this, it is meant to allow for zero degree-in nodes (in case every
        # other node is too far)
        self.G = dgl.add_self_loop(self.G)
        
    def update(self, d_x, d_y, d_data):
        # GUY TODO: add some "physics" here. meaning, if they are too close, maybe
        # just get them to be tangent.
        for i in range( self.n ):
            self.robots[i].x    += d_x[i].detach().numpy()
            if(self.robots[i].x > self.fsize[0]):
                self.robots[i].x = self.fsize[0] - 0.01 #just so there won't be indexing prob.
            if(self.robots[i].x < 0):
                self.robots[i].x = 0
            self.robots[i].y    += d_y[i].detach().numpy()
            if(self.robots[i].y > self.fsize[1]):
                self.robots[i].y = self.fsize[1] - 0.01 #just so there won't be indexing prob.
            if(self.robots[i].y < 0):
                self.robots[i].y = 0
            self.robots[i].data += d_data[i].detach().numpy()
            
        self.create_graph(first_time=False)
        
    def get_pose_state(self):
        state = th.zeros(self.fsize[0], self.fsize[1], requires_grad=True)
        for r in self.robots:
            # if it is in a cell of 1x1, mark it
            state[int(r.x), int(r.y)] = 1
        return state
        
if __name__=='__main__':
    w = Simulator(num_robots=5, memory_size=10, range_len=5.)
        
        