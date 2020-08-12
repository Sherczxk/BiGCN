from sklearn.metrics import roc_auc_score
import argparse
from model import BiGCN
from utils import *
from dataset import *
import torch.nn as nn
import copy

torch.cuda.set_device(0)

if not os.path.isdir('results'):
    os.mkdir('results')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--cases', dest='cases', default='all', type=str,
                    help='sample_rate,noise_rate,noise_level')
parser.add_argument('--dataset', dest='dataset', default='dblp', type=str,help='Cora;Citeseer;Pubmed; dblp')

parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)
parser.add_argument('--permute', dest='permute', action='store_true',
                    help='whether permute subsets')
parser.add_argument('--dropout',dest='dropout',type=float,default=0.5,help='Dropout rate')
parser.add_argument('--approximate', dest='approximate', default=-1, type=int,
                    help='k-hop shortest path distance. -1 means exact shortest path') # -1, 2
parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=8, type=int) # implemented via accumulating gradient
parser.add_argument('--hidden_dim', dest='hidden_dim', default=32, type=int)
parser.add_argument('--output_dim', dest='output_dim', default=32, type=int)
parser.add_argument('--anchor_num', dest='anchor_num', default=64, type=int)


parser.add_argument('--lam',type=float,default=1,help='L1 norm weight')
parser.add_argument('--lr', dest='lr', default=0.01, type=float)
parser.add_argument('--epoch_num', dest='epoch_num', default=100, type=int)
parser.add_argument('--repeat_num', dest='repeat_num', default=1, type=int) # 10
parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)
parser.add_argument('--A2_type',dest='A2_type',type=str,default='learn_A2',help='Type of A2, cos_A2,cosM_A2,learn_A2')
parser.add_argument('--Type',dest='Type',type=str,default='mean',help='Forward type of Y in ADMM')
parser.add_argument('--sample_rate', dest='sample_rate', default=1, type=int)
parser.add_argument('--noise_rate', dest='noise_rate', default=0, type=int)
parser.add_argument('--noise_level', dest='noise_level', default=0, type=int)
parser.add_argument('--p', dest='p', type=int)
args = parser.parse_args(args=[])

def link_predict(args):  
    seeds = [5,10,15,20,25,30,35,40,45,50]
    sample_rate = [1,0.8,0.6,0.4,0.2]
    noise_rate = [0.2,0.4,0.6,0.8,1]
    noise_level = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    
    dataset_name = args.dataset
    if args.cases == 'all':
        case = ['sample_rate','noise_rate','noise_level']
    else:
        case = args.cases    
    
    print('Begin: BiGCN, data:%s'%(dataset_name))
    time1 = time.time()
    data_list = get_tg_dataset(args, dataset_name)
    time2 = time.time()
    print(dataset_name, 'load time',  time2-time1)
    num_features = data_list[0].x.shape[1]
    num_node_classes = None
    num_graph_classes = None
    if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
        num_node_classes = max([data.y.max().item() for data in data_list])+1
    if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
        num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
    print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
    nodes = [data.num_nodes for data in data_list]
    edges = [data.num_edges for data in data_list]
    print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
    print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

    args.batch_size = min(args.batch_size, len(data_list))
    print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))


    def train(data_l):

        # model
        input_dim = data_l[0].x.shape[1]
        output_dim = args.output_dim

        model = BiGCN(input_dim, args.hidden_dim,args.output_dim, args.p,args.lambda_1,args.lambda_2, args.dropout,bias=True,beta=True,A2=args.A2_type,n_iter=2,Type=args.Type).to(device)
        # loss
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        loss_func = nn.BCEWithLogitsLoss()
        out_act = nn.Sigmoid()
        result_val = []
        result_test = []

        for epoch in range(args.epoch_num):
            if epoch==200:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
            model.train()
            optimizer.zero_grad()
            shuffle(data_l)
            effective_len = len(data_l)//args.batch_size*len(data_l)
            for id, data in enumerate(data_l[:effective_len]):
                if args.permute:
                    preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                out,A = model(data)
                edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0,:]).long().to(device))
                nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1,:]).long().to(device))
                pred = torch.sum(nodes_first * nodes_second, dim=-1)
                label_positive = torch.ones([data.mask_link_positive_train.shape[1],], dtype=pred.dtype)
                label_negative = torch.zeros([data.mask_link_negative_train.shape[1],], dtype=pred.dtype)
                label = torch.cat((label_positive,label_negative)).to(device)
                if args.A2_type =='learn_A2':
                    loss = loss_func(pred, label)+args.lam*(torch.norm(A[0], p=1)+torch.norm(A[1], p=1))
                else:
                    loss = loss_func(pred, label)
                # update
                loss.backward()
                if id % args.batch_size == args.batch_size-1:
                    if args.batch_size>1:
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad /= args.batch_size
                    optimizer.step()
                    optimizer.zero_grad()


            if epoch % args.epoch_log == 0:
                # evaluate
                model.eval()
                loss_train = 0
                loss_val = 0
                loss_test = 0
                correct_train = 0
                all_train = 0
                correct_val = 0
                all_val = 0
                correct_test = 0
                all_test = 0
                auc_train = 0
                auc_val = 0
                auc_test = 0
                emb_norm_min = 0
                emb_norm_max = 0
                emb_norm_mean = 0
                for id, data in enumerate(data_l):
                    out,_ = model(data)
                    emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                    emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                    emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()


                    edge_mask_train = np.concatenate((data.mask_link_positive_train, data.mask_link_negative_train), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_train[1, :]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones([data.mask_link_positive_train.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_train.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)
                    loss_train += loss_func(pred, label).cpu().data.numpy()
                    auc_train += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                    # val
                    edge_mask_val = np.concatenate((data.mask_link_positive_val, data.mask_link_negative_val), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_val[1, :]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones([data.mask_link_positive_val.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_val.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)
                    loss_val += loss_func(pred, label).cpu().data.numpy()
                    auc_val += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())
                    # test
                    edge_mask_test = np.concatenate((data.mask_link_positive_test, data.mask_link_negative_test), axis=-1)
                    nodes_first = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[0, :]).long().to(device))
                    nodes_second = torch.index_select(out, 0, torch.from_numpy(edge_mask_test[1, :]).long().to(device))
                    pred = torch.sum(nodes_first * nodes_second, dim=-1)
                    label_positive = torch.ones([data.mask_link_positive_test.shape[1], ], dtype=pred.dtype)
                    label_negative = torch.zeros([data.mask_link_negative_test.shape[1], ], dtype=pred.dtype)
                    label = torch.cat((label_positive, label_negative)).to(device)
                    loss_test += loss_func(pred, label).cpu().data.numpy()
                    auc_test += roc_auc_score(label.flatten().cpu().numpy(), out_act(pred).flatten().data.cpu().numpy())

                loss_train /= id+1
                loss_val /= id+1
                loss_test /= id+1
                emb_norm_min /= id+1
                emb_norm_max /= id+1
                emb_norm_mean /= id+1
                auc_train /= id+1
                auc_val /= id+1
                auc_test /= id+1

                result_val.append(auc_val)
                result_test.append(auc_test)

        result_val = np.array(result_val)
        result_test = np.array(result_test)
        results = result_test[np.argmax(result_val)]
        return results       


    print('-------------------------data:%s,model:BiGCN-----------------------'%(dataset_name))

    if 'sample_rate' in case:
        acc1 = []
        args.lambda_1 = 1.2
        args.lambda_2 = 1.2
        args.p = 8.5
        for sample_r in sample_rate:
            noise_r = 0
            noise_l = 0 
            results = []

            for seed in seeds:
                set_seed(seed)
                data_l = copy.deepcopy(data_list)
                for i,data in enumerate(data_l):
                    preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cuda')
                    data = get_noise_data(data,sample_r,noise_r,noise_l)
                    data = data.to(device)
                    data_l[i] = data
                result = train(data_l)
                print(result)
                results.append(result)
            results = np.array(results)
            results_mean = np.mean(results).round(6)
            results_std = np.std(results).round(6)
            acc1.append(results_mean) 
            print('sample_rate:%f,results:%f,std:%f'%(sample_r,results_mean,results_std))
            print(results)
        print(args.p,args.lambda_1,args.Type,'sample_rate,results:')
        print(acc1,'---------')

        
    if 'noise_rate' in case:
        acc2 = [] 
        args.lambda_1 = 1.2
        args.lambda_2 = 1.2
        args.p = 8.5
        for noise_r in noise_rate:
            sample_r = 1
            noise_l = 0

            results = []
            for seed in seeds:
                set_seed(seed)
                data_l = copy.deepcopy(data_list)
                for i,data in enumerate(data_l):
                    preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cuda')
                    data = get_noise_data(data,sample_r,noise_r,noise_l)
                    data = data.to(device)
                    data_l[i] = data
                result = train(data_l)
                print(result)
                results.append(result)
            results = np.array(results)
            results_mean = np.mean(results).round(6)
            results_std = np.std(results).round(6)
            acc2.append(results_mean) 
            print('noise_rate:%f,results:%f'%(noise_r,results_mean))
            print(results)
        print(args.p,args.lambda_1,args.Type,'noise_rate,results:')
        print(acc2,'-----')


    if 'noise_level'in case:
        acc3 = []
        args.lambda_1 = 1.2
        args.lambda_2 = 1.2
        args.p = 8.5
        for noise_l in noise_level:
            sample_r = 1
            noise_r = 1

            results = []
            for seed in seeds:
                set_seed(seed)
                data_l = copy.deepcopy(data_list)
                for i,data in enumerate(data_l):
                    preselect_anchor(data, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cuda')
                    data = get_noise_data(data,sample_r,noise_r,noise_l)
                    data = data.to(device)
                    data_l[i] = data
                result = train(data_l)
                print(result)
                results.append(result)
            results = np.array(results)
            results_mean = np.mean(results).round(6)
            results_std = np.std(results).round(6)
            acc3.append(results_mean) 
            print('noise_level:%f,results:%f'%(noise_l,results_mean))
            print(results)
        print(args.p,args.lambda_1,args.Type,'noise_level,results:')
        print(acc3,'------')

    print('\n')
    print('-----------------------BiGCN ',dataset_name,args.p,args.lambda_1,args.Type, 'Final Result:--------------------')
    if 'sample_rate' in case:
        print('sample_rate:\n',acc1)
    if 'noise_rate' in case:
        print('noise_rate:\n',acc2)
    if 'noise_level' in case:
        print('noise_level:\n',acc3)
    print('----------------------------------finish--------------------------------------------------------')

print(args)
link_predict(args)
