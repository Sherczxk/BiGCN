from torch_geometric.data import Data, DataLoader
import torch
import torch_geometric as tg
import torch_geometric.datasets
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
import numpy as np
import scipy.sparse as sp
import sys
import os
import pickle as pkl
import networkx as nx
from time import perf_counter
from torch.utils import data
import random
import scipy
import math
from torch_geometric.data import Data
from utils import Normalize
import torch_geometric.transforms as T
#torch.cuda.set_device(1)
path = os.path.expanduser("./data/")
label_rate = {'Cora':0.052,'Citeseer':0.036,'Pubmed':0.003,'Computers':0.015,'Photo':0.021}

Norm = Normalize()

def get_dataset(name):
    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(path+name, name)
    elif name in ['Computers', 'Photo']:
        dataset = Amazon(path+name, name)
    else:
        raise Exception('Unknown dataset.')
    return dataset


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def adj_to_edge_index(adj):
    r1 = []
    r2 = []
    for i,k in enumerate(adj.tocsc()):
        for j in list(k.indices):
            r1.append(i)
            r2.append(j)
    edge_index = torch.LongTensor([r1,r2])
    return edge_index

def load_dataset(dataset_str="Cora",sample_rate=1,noise_rate=0,noise_level=0,adj_mistake=0,norm_type='SymNorm_tildeA',cuda=True, shuffle=False):
    """
    Load  Datasets.
    """
    label_r = label_rate[dataset_str]
    dataset = get_dataset(dataset_str)

    features = dataset.data.x.numpy()
    labels = dataset.data.y
    e = dataset.data.edge_index
    data = np.array([1]*len(e[0]))
    adj = sp.csr_matrix((data, (e[0].numpy(), e[1].numpy())))
    
    
    # missing some features
    if sample_rate<1:
        num_features = features.shape[1]
        feature_index = list(range(num_features))
        sample_index = round(num_features*sample_rate)
        feature_sample = random.sample(feature_index,sample_index)
        feature_sample.sort()
        tmp_matrix = features.T
        tmp_matrix = tmp_matrix[feature_sample]
        features = tmp_matrix.T
    
    
    #adding noise to features
    if noise_rate>0:
        num_node= features.shape[0]
        node_idex = list(range(num_node))
        noise_num = round(num_node*noise_rate)
        node_noise_idex = random.sample(node_idex,noise_num)
        if noise_level>0:
            varians = np.array([noise_level]*noise_num)
        else:
            varians = np.random.rand(noise_num)
        for v,i in enumerate(node_noise_idex):
            noise = np.random.normal(0,varians[v],size=features.shape[1]) 
            features[i] = features[i] + noise
            
            
    # adjacent mistake
    if adj_mistake>0:
        M = torch.FloatTensor(adj.shape[0],adj.shape[1]).uniform_() > adj_mistake
        M = torch.triu(M.float(),diagonal=1)
        M = M + M.t()
        e = torch.ones(adj.shape)
        A = torch.FloatTensor(np.array(adj.todense()))
        adj = A*M+(e-A)*(e-M)
        adj = scipy.sparse.csr_matrix(adj.cpu().numpy())
        
    edge_index = adj_to_edge_index(adj)
    
    adj = Norm(adj,norm_type)
    if dataset_str in [ 'Citeseer', 'Pubmed']:
        train_mask = dataset.data.train_mask
        test_mask = dataset.data.test_mask
        val_mask = dataset.data.val_mask
        nidx_test = []
        nidx_val = []
        nidx_train = []
        for idx, mask in enumerate(train_mask):
            if mask:
                nidx_train.append(idx)
        for idx, mask in enumerate(test_mask):
            if mask:
                nidx_test.append(idx)
        for idx, mask in enumerate(val_mask):
            if mask:
                nidx_val.append(idx)
    else:
        ids = [*range(int(len(labels)*label_r)+1500)]
        if shuffle:
            random.shuffle(ids)
        nidx_test = ids[-1000:]
        nidx_val = ids[int(len(labels)*label_r):int(len(labels)*label_r)+500]
        nidx_train = ids[:int(len(labels)*label_r)]

        train_mask = [False]*len(labels)
        val_mask = [False]*len(labels)
        test_mask = [False]*len(labels)
        for i in nidx_test:
            test_mask[i] = True
        for i in nidx_val:
            val_mask[i] = True
        for i in nidx_train:
            train_mask[i] = True
    
    
    features = torch.FloatTensor(features).float()
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    

    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = torch.tensor(nidx_train).cuda()
        idx_val = torch.tensor(nidx_val).cuda()
        idx_test = torch.tensor(nidx_test).cuda()
    data = Data(x=features,y=labels,edge_index=edge_index)
    if dataset_str in ['Citeseer', 'Pubmed']:
        data.train_mask = dataset[0].train_mask
        data.val_mask = dataset[0].val_mask
        data.test_mask = dataset[0].test_mask
    else:
        data.train_mask = torch.tensor(train_mask)
        data.val_mask = torch.tensor(val_mask)
        data.test_mask = torch.tensor(test_mask)

    return adj, features, labels,idx_train, idx_val, idx_test, data



