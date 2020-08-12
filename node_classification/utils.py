import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from scipy.linalg import solve_banded

class Normalize(Module):
    def __init__(self):
        super(Normalize,self).__init__()
        
    def sym_normalized_tildeA(self,adj):
        '''
        \tildeD^{-1/2}\tildeA\tildeD^{-1/2}   #GCN
        '''
        adj = adj + sp.eye(adj.shape[0])  #\tildeA
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten() #\tildeD^{-1/2}
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()   #\tildeD^{-1/2}\tildeA\tildeD^{-1/2}
    
    def row_normalized_tildeA(self,adj): 
        '''
        \tildeD^{-1}\tildeA
        '''
        adj = adj + sp.eye(adj.shape[0])    #\tildeA
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_inv = np.power(row_sum, -1).flatten()   #\tildeD^{-1}
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        return d_mat_inv.dot(adj).tocoo()      #\tildeD^{-1}\tildeA

    def sym_normalized_L(self,adj): 
        '''
        D^{-1/2}LD^{-1/2}=I-D^{-1/2}AD^{-1/2}    #BiGCN
        '''
        adj = sp.coo_matrix(adj)
        row_sum = np.array(adj.sum(1))
        d_mat = sp.diags(row_sum.flatten())     #D
        L = d_mat - adj                  #L=D-A
        d_inv = np.power(row_sum+1e-8, -1/2).flatten() #D^{-1/2}
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)          
        mx = d_mat_inv.dot(L).dot(d_mat_inv).tocoo()

        return mx  #D^{-1/2}LD^{-1/2}


    def forward(self,mx,type,alpha=0.8,scale=False):
        if scale:
            mx = self.rescale_laplacian(mx,scale)
        elif type == 'SymNorm_tildeA':
            return self.sym_normalized_tildeA(mx)        
        elif type == 'LeftNorm_tildeA':
            return self.row_normalized_tildeA(mx)
        elif type == 'SymNorm_L':
            return self.sym_normalized_L(mx)            #BiGCN
        elif type == 'AlphaNorm_L':
            return self.inv_normalized_L(mx,alpha)
        else:
            raise Expection("Invalid normalization technique:", type)
    



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)



def set_seed(seed, cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

