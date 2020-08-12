import time
start_time = time.time()
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from dataloader import load_dataset
from utils import accuracy, set_seed
from model import BiGCN
import csv
import os
import warnings
warnings.filterwarnings("ignore")

#torch.cuda.set_device(4)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs',type=int,default=10000,help='Number of epochs to train')
parser.add_argument('--dataset',type=str,default='Cora', help='Cora,Citeseer,Pubmed,Computers,Photo')
parser.add_argument('--cases',type=str,default='sample_rate', help='sample_rate,noise_rate,noise_level,adjacent_mistake')
parser.add_argument('--lr',type=float,default=0.01,help='Initial learning rate,adam:0.001,sgd:0.05')
parser.add_argument('--weight_decay',type=float,default=0.007,help='weight decay') 
parser.add_argument('--hidden',type=int,default=16,help='Number of hidden units')
parser.add_argument('--dropout',type=float,default=0.5,help='Dropout rate')

parser.add_argument('--lam',type=float,default=1,help='L1 norm weight')
parser.add_argument('--n_iter',type=int,default=2,help='iteration times of ADMM')
parser.add_argument('--optimize_type',type=str,default='Adam', help='Optimization method, Adam; SGD')
parser.add_argument('--A2_type',type=str,default='learn_A2',help='Method of generate A2:cos_A2,cosM_A2,learn_A2')
parser.add_argument('--norm_type',type=str,default='SymNorm_L',help= 'Normalization method:SymNorm_tildeA, LeftNorm_tildeA, SymNorm_L')
parser.add_argument('--Type',type=str,default='mean',help='Forward type of Y in ADMM')
args = parser.parse_args()

print(args)
###########################################################################################################################
def node_classification( seed, dataset, sample_rate, noise_rate, noise_level,
              adj_mistake, norm_type, optimize_type, p,lambda_1,lambda_2,Type,
              A2_type, dropout, hidden, weight_decay, lr, epochs,n_iter,lam):
    set_seed( seed, cuda=True)
    L, features, labels,idx_train, idx_val, idx_test,data=load_dataset(dataset,sample_rate,noise_rate,noise_level,adj_mistake,norm_type,cuda=True, shuffle=False)
    
    I_1 = torch.eye(L.shape[0]).cuda()
    L = I_1-lambda_1*L
    
    num_feature = features.shape[1]
    #print('num_feature: ',features.shape[1])
    num_class = labels.max().item()+1
    #print('num_class',num_class)

    model = BiGCN(num_feature, hidden,num_class, p,lambda_1,lambda_2, dropout,bias=True,beta=True,A2= A2_type,n_iter=n_iter,Type=Type).cuda()

    if  optimize_type == "Adam":
        optimizer = optim.Adam(model.parameters(),lr= lr,weight_decay= weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),lr= lr,weight_decay= weight_decay)


    def train():
        start = time.time()
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output,A2 = model(features,L)
        if A2_type == 'learn_A2':
            loss_train = F.nll_loss(output[idx_train],labels[idx_train])+lam*torch.norm(A2[0], p=1)
        else:
            loss_train = F.nll_loss(output[idx_train],labels[idx_train])

        acc_train = accuracy(output[idx_train],labels[idx_train])
        loss_train.backward()
        optimizer.step()
          
    def test():
        model.eval()
        output,_ = model(features,L)
        acc_val = accuracy(output[idx_val],labels[idx_val])
        acc_test = accuracy(output[idx_test],labels[idx_test])
        return acc_val.item(),acc_test.item()

    acc = []
    best_dict = {
        "val_acc":0,
        "test_acc":0,
        "epoch":-1
    }
    patience = 100
    patience_counter = 0
    for i in range(epochs):
        if patience_counter >= patience: 
            break
        train()
        val_acc,test_acc = test()
        #print("epoch",i, "val acc:", val_acc, "test_acc", test_acc)
        if val_acc < best_dict["val_acc"]:
            patience_counter = patience_counter + 1
        else:
            best_dict["val_acc"] = val_acc
            best_dict["test_acc"] = test_acc
            best_dict["epoch"] = i
            patience_counter = 0
    print("finish----------")
    print(best_dict)
    return best_dict["test_acc"]

##############################################################################################################################

random_seed = [5,10,15,20,25,30,35,40,45,50]
sample_rate = [1,0.8,0.6,0.4,0.2]
noise_rate = [0.2,0.4,0.6,0.8,1]
noise_level = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
adj_mistake1 = [0.001,0.003,0.005,0.007,0.009,0.011,0.013,0.015]
adj_mistake2 = [1e-5,2.5e-5,5e-5,7.5e-5,1e-4,2.5e-4,5e-4,7.5e-4,1e-3]



logfilename = "log/exp_result_"+str(time.time())+".csv"
logcsvfile = open(logfilename, "w", newline="")
logwriter = csv.writer(logcsvfile)
logwriter.writerow(["id", "dataset","model", "p", "lambda", "L1_norm", "Y_type","case", "sample_rate","noise_rate","noise_level", "adj_mistake", "run_time", "test_acc", "test_std", "timing"])
logcsvfile.flush()
id_count = 1
def log_write(dataset, model, p, lamb, L1_norm, Y_type, case, sample_rate, noise_rate, noise_level, adj_mistake, run_time, test_acc, test_std, timing):
    global id_count
    logwriter.writerow([id_count, dataset, model, p, lamb, L1_norm, Y_type, case, sample_rate, noise_rate, noise_level, adj_mistake, run_time, test_acc, test_std, timing])
    logcsvfile.flush()
    id_count = id_count + 1

if args.cases == 'all':
    case = ['sample_rate','noise_rate','noise_level','adjacent_mistake']
else:
    case = args.cases
if args.dataset == 'Pubmed':
    adj_mistake = adj_mistake2
else:
    adj_mistake = adj_mistake1
print('data:%s,model:BiGCN, norm_type:%s, A2_type:%s'%(args.dataset,args.norm_type,args.A2_type))


if 'sample_rate' in case:
    local_case = 'sample_rate'
    mark1 = []
    args.lambda_1 = 1.8
    args.lambda_2 = 1.8
    args.p = 3
    print("sample_rate")
    print(args)
    for sample in sample_rate:
        args.sample_rate = sample
        args.noise_rate = 0
        args.noise_level = 0
        args.adj_mistake = 0
        acc = []
        local_start_time = time.time()
        for i,seed in enumerate(random_seed):
            args.random_seed = seed
            t = node_classification(args.random_seed,args.dataset,args.sample_rate,args.noise_rate,args.noise_level,
             args.adj_mistake,args.norm_type,args.optimize_type,args.p,args.lambda_1,args.lambda_2,args.Type,
             args.A2_type,args.dropout,args.hidden,args.weight_decay,args.lr,args.epochs,args.n_iter,args.lam)
            acc.append(t)
        duration = time.time() - local_start_time
        print('model:BiGCN, data:%s, sample_rate:%f'%(args.dataset,sample))
        print(acc)
        print('mean:%f std:%f' %(np.mean(acc), np.std(acc)))
        test_acc = np.mean(acc)
        test_std = np.std(acc)
        mark1.append(np.mean(acc))
        log_write(args.dataset, "BiGCN", args.p, args.lambda_1, args.lam, args.Type,local_case,args.sample_rate, args.noise_rate, args.noise_level, args.adj_mistake, len(random_seed), test_acc, test_std, duration) 
    print('model:BiGCN, data:%s, sample_rate'%(args.dataset))

    print(mark1)
    
if 'noise_rate' in case:
    mark2 = []
    local_case = 'noise_rate'
    args.lambda_1 = 1.8
    args.lambda_2 = 1.8
    args.p = 3
    for noise_r in noise_rate:
        args.noise_rate = noise_r
        args.sample_rate = 1
        args.noise_level = 0
        args.adj_mistake = 0
        acc = []
        local_start_time = time.time()
        for i,seed in enumerate(random_seed):
            args.random_seed = seed
            t = node_classification(args.random_seed,args.dataset,args.sample_rate,args.noise_rate,args.noise_level,
             args.adj_mistake,args.norm_type,args.optimize_type,args.p,args.lambda_1,args.lambda_2,args.Type,
             args.A2_type,args.dropout,args.hidden,args.weight_decay,args.lr,args.epochs,args.n_iter,args.lam)
            acc.append(t)
        duration = time.time() - local_start_time
        print('model:BiGCN, data:%s, noise_rate:%f'%(args.dataset,noise_r))
        print(acc)
        print('mean:%f std:%f' %(np.mean(acc), np.std(acc)))
        mark2.append(np.mean(acc))
        test_acc = np.mean(acc)
        test_std = np.std(acc)
        log_write(args.dataset, "BiGCN", args.p, args.lambda_1, args.lam, args.Type,local_case,args.sample_rate, args.noise_rate, args.noise_level, args.adj_mistake, len(random_seed), test_acc, test_std, duration)
    print('model:BiGCN, data:%s, noise_rate'%(args.dataset))
    print(mark2)

    
if 'noise_level' in case:
    mark3 = []
    args.lambda_1 = 1.8
    args.lambda_2 = 1.8
    args.p = 3
    local_case = 'noise_level'
    for noise_l in noise_level:
        args.noise_rate = 1
        args.sample_rate = 1
        args.noise_level = noise_l
        args.adj_mistake = 0
        acc = []
        local_start_time = time.time()
        for i,seed in enumerate(random_seed):
            args.random_seed = seed
            t = node_classification(args.random_seed,args.dataset,args.sample_rate,args.noise_rate,args.noise_level,
             args.adj_mistake,args.norm_type,args.optimize_type,args.p,args.lambda_1,args.lambda_2,args.Type,
             args.A2_type,args.dropout,args.hidden,args.weight_decay,args.lr,args.epochs,args.n_iter,args.lam)
            acc.append(t)
        duration = time.time() - local_start_time
        print('model:BiGCN, data:%s, noise_level:%f'%(args.dataset,noise_l))
        print(acc)
        print('mean:%f std:%f' %(np.mean(acc), np.std(acc)))
        mark3.append(np.mean(acc))
        test_acc = np.mean(acc)
        test_std = np.std(acc)
        log_write(args.dataset, "BiGCN", args.p, args.lambda_1, args.lam, args.Type,local_case,args.sample_rate, args.noise_rate, args.noise_level, args.adj_mistake, len(random_seed), test_acc, test_std, duration)
    print('model:BiGCN, data:%s, noise_level'%(args.dataset))
    print(mark3)

    
if 'adjacent_mistake' in case:    
    mark4 = [] 
    local_case = 'adjacent_mistake'
    args.lambda_1 = 0.8
    args.lambda_2 = 0.8
    args.p = 0.1
    print("adjacent_mistake")
    print(args)
    for adj in adj_mistake:
        args.noise_rate = 0
        args.sample_rate = 1
        args.noise_level = 0
        args.adj_mistake = adj
        acc = []
        local_start_time = time.time()
        for i,seed in enumerate(random_seed):
            args.random_seed = seed
            t = node_classification(args.random_seed,args.dataset,args.sample_rate,args.noise_rate,args.noise_level,
             args.adj_mistake,args.norm_type,args.optimize_type,args.p,args.lambda_1,args.lambda_2,args.Type,
             args.A2_type,args.dropout,args.hidden,args.weight_decay,args.lr,args.epochs,args.n_iter,args.lam)
            acc.append(t)
        duration = time.time() - local_start_time
        print('model:BiGCN, data:%s, adj_mistake:%f'%(args.dataset,adj))
        print(acc)
        print('mean:%f std:%f' %(np.mean(acc), np.std(acc)))
        test_acc = np.mean(acc)
        test_std = np.std(acc)
        mark4.append(np.mean(acc))
        log_write(args.dataset, "BiGCN", args.p, args.lambda_1, args.lam, args.Type,local_case,args.sample_rate, args.noise_rate, args.noise_level, args.adj_mistake, len(random_seed), test_acc, test_std, duration)
    print('model:BiGCN, data:%s, adj_mistake'%(args.dataset))
    print(mark4)

print('\n')
print('---------------------------------- BiGCN',args.dataset,'Final Result-------------------------------------')
if 'sample_rate' in case:
    print('sample_rate:\n',mark1)
if 'noise_rate' in case:
    print('noise_rate:\n',mark2)
if 'noise_level' in case:
    print('noise_level:\n',mark3)
if 'adjacent_mistake' in case:   
    print('adj_mistake:\n',mark4)
print("total time", time.time() - start_time)
logcsvfile.close()