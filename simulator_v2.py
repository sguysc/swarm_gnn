# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:53:31 2021

@author: guy
"""

import numpy as np
import networkx as nx
import dgl
import torch as th
import PIL.Image
# from dgl.data import DGLDataset

# class MyDataset(DGLDataset):
#     _fname = 'pic1.png'
#     _max_size = 40
#     """ Template for customizing graph datasets in DGL.

#     Parameters
#     ----------
#     url : str
#         URL to download the raw dataset
#     raw_dir : str
#         Specifying the directory that will store the
#         downloaded data or the directory that
#         already stores the input data.
#         Default: ~/.dgl/
#     save_dir : str
#         Directory to save the processed dataset.
#         Default: the value of `raw_dir`
#     force_reload : bool
#         Whether to reload the dataset. Default: False
#     verbose : bool
#         Whether to print out progress information
#     """
#     def __init__(self,
#                   url=None,
#                   raw_dir=None,
#                   save_dir=None,
#                   force_reload=False,
#                   verbose=False):
#         super(MyDataset, self).__init__(name='dataset_name',
#                                         url=url,
#                                         raw_dir=raw_dir,
#                                         save_dir=save_dir,
#                                         force_reload=force_reload,
#                                         verbose=verbose)

#     def download(self):
#         # download raw data to local disk
#         pass

#     def process(self):
#         # process data to a list of graphs and a list of labels
#         self.ref_img = PIL.Image.open(self._fname)
#         self.ref_img.thumbnail((self._max_size, self._max_size), PIL.Image.ANTIALIAS)
#         self.ref_img = 1.0-th.float32(self.ref_img)
#         self.number_of_robots = self.ref_img.sum().int()

#         self.graphs, self.label = self._load_graph(mat_path)

#     def __getitem__(self, idx):
#         # get one example by index
#         pass

#     def __len__(self):
#         # number of data examples
#         pass

#     def save(self):
#         # save processed data to directory `self.save_path`
#         pass

#     def load(self):
#         # load processed data from directory `self.save_path`
#         pass

#     def has_cache(self):
#         # check whether there are processed data in `self.save_path`
#         pass
        
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
        breakpoint()
        # get all current locations.
        # X = self.robots[:, 1:3]
        # dist_sq = th.sqrt(th.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = -1))
        # dist_sq[dist_sq > self.range_len] = 0
        # self.G = dgl.from_networkx(dist_sq)
        # if(first_time):
        X = self.robots[:, 1:3].detach().numpy()
        # create a distance matrix
        dist_sq = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis = -1))
        # clear all the connections from robots that are too far away
        dist_sq[dist_sq > self.range_len] = 0.
        # create the graph connections
        self.Gnx = nx.from_numpy_matrix(dist_sq)
        self.G = dgl.from_networkx(self.Gnx)
        # else:
            
        
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
        
        