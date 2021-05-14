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
    def __init__(self, num_robots=10, memory_size=10, range_len=2., max_x=10, max_y=10):

        self.robots = th.zeros(num_robots, 1+2+memory_size)
        self.n = num_robots
        self.range_len = range_len
        self.memory_size = memory_size
        self.fsize = (max_x, max_y)
        for rid in range(num_robots):
            # sample an initial location for this robot. the -0.1 is so we don't have a robot on the max index
            self.robots[rid,0] = rid
            self.robots[rid,1] = np.around((max_x-0.1)*np.random.rand(), decimals=1)
            self.robots[rid,2] = np.around((max_y-0.1)*np.random.rand(), decimals=1)
        
        self.map = th.zeros(max_x, max_y)
        self.create_graph()
            
    def create_graph(self, first_time=True):
        # get all current locations.
        # X = [ [r.x, r.y] for r in self.robots]
        # X = np.array(X)
        X = self.robots[:, 1:3].detach().numpy()
        
        # create a distance matrix
        dist_sq = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = -1))
        # clear all the connections from robots that are too far away
        dist_sq[dist_sq > self.range_len] = 0.
        # create the graph connections
        self.Gnx = nx.from_numpy_matrix(dist_sq)
        self.G = dgl.from_networkx(self.Gnx)
        
        # add the features to the nodes and edges
        self.G.ndata['x']     = self.robots[:, 1]
        self.G.ndata['y']     = self.robots[:, 2]
        # data is essentially all we throw to the neural network. we don't want
        # the nn to learn things using the position / id
        # if(first_time):
        self.G.ndata['data']  = self.robots[:, 3:]
        # else:
        #     data = [ [r.data.tolist() ] for r in self.robots]
        #     data = np.array(data).squeeze()
        #     self.G.ndata['data']  = th.tensor(data)
            
        # self.G.ndata['id']    = self.robots[:, 0]
        #th.linspace(0, self.G.num_nodes()-1, self.G.num_nodes(), dtype=th.int32).reshape(self.G.num_nodes(),-1)
        # GUY TODO: get the true ranges. for now, we treat all as equal
        self.G.edata['range'] = th.ones(self.G.num_edges(), 1)
        # GUY TODO: check this, it is meant to allow for zero degree-in nodes (in case every
        # other node is too far)
        self.G = dgl.add_self_loop(self.G)
        
    def update(self, d_x, d_y, d_data):
        # GUY TODO: add some "physics" here. meaning, if they are too close, maybe
        # just get them to be tangent.
        self.robots[:, 1] += d_x
        self.robots[self.robots[:, 1] > self.fsize[0], : ] = self.fsize[0]
        self.robots[self.robots[:, 1] < 0, : ] = 0
        self.robots[:, 2] += d_y
        self.robots[self.robots[:, 2] > self.fsize[1], : ] = self.fsize[1]
        self.robots[self.robots[:, 2] < 0, : ] = 0
        self.robots[:, 3:] += d_data
            
        self.create_graph(first_time=False)
        
    def get_pose_state(self):
        self.map *= 0.
        one = th.ones(1, dtype=th.int)
        for r in range(self.n):
            # if it is in a cell of 1x1, mark it
            self.map[self.robots[r, 1].int(), self.robots[r, 2].int()] = one 
        return self.map
        
if __name__=='__main__':
    w = Simulator(num_robots=5, memory_size=10, range_len=5.)
        
        