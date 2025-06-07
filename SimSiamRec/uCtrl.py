# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 10:41:57 2020

@author: ZAR
"""

import torch
from torch import nn
import scipy.sparse as sp
import random
import numpy as np
import logging
import torch.optim as optim
import os
from time import strftime
from time import localtime
from torch.utils.data import DataLoader, Dataset
import argparse
import reckit
import sys
from tqdm import tqdm
from RankingMetrics import *


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class uCtrl_train_dataset(Dataset):
    def __init__(self, 
                 user_list, 
                 item_list):
        super(Dataset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        
    def __getitem__(self, index):
        return self.user_list[index], self.item_list[index]
               
    def __len__(self):
        return len(self.user_list) 
    
class uCtrl(nn.Module):
    def __init__(self, data_train, num_users, num_items):
        super(uCtrl, self).__init__()
        self.train_data = data_train
        self.num_users  = num_users  
        self.num_items  = num_items   
        self.embedding_size = args.embedding_size
        self.layers = args.layer  
        
        self.embedding_user = nn.Embedding(num_users, self.embedding_size)
        self.embedding_item = nn.Embedding(num_items, self.embedding_size)
        
        self.embedding_user_final = None
        self.embedding_item_final = None

        nn.init.normal_(self.embedding_user.weight, 0,0.01)
        nn.init.normal_(self.embedding_item.weight, 0,0.01)
        
        self.users_pop, self.items_pop = get_ui_pop(self.train_data, self.num_users, self.num_items)
        self.users_pop, self.items_pop = self.users_pop, self.items_pop
        
        self.projection_u  = torch.empty((2, self.embedding_size, self.embedding_size))
        self.projection_u  = torch.nn.Parameter(self.projection_u, requires_grad=True)
        nn.init.xavier_normal_(self.projection_u)
        
        self.projection_i  = torch.empty((2, self.embedding_size, self.embedding_size))
        self.projection_i  = torch.nn.Parameter(self.projection_i, requires_grad=True)
        nn.init.xavier_normal_(self.projection_i) 
        
        self.Graph_ui = self.getSparseGraph()
        
    def forward(self, users, items):
        all_users, all_items = self.computer()
        self.embedding_user_final = all_users.detach()
        self.embedding_item_final = all_items.detach()
        
        users_emb = all_users[users]
        items_emb = all_items[items]
        
        users_emb = nn.functional.normalize(users_emb,dim=-1)
        items_emb = nn.functional.normalize(items_emb,dim=-1)
        
        users_relation_emb = torch.einsum("ik,ikj->ij", [users_emb.detach(), self.projection_u[self.users_pop[users] ]])
        items_relation_emb = torch.einsum("ik,ikj->ij", [items_emb.detach(), self.projection_i[self.items_pop[items] ]])
        
        users_relation_emb = nn.functional.normalize(users_relation_emb,dim=-1)
        users_relation_emb = nn.functional.normalize(users_relation_emb,dim=-1)
        
        o_space=True
        align_relation = self.lalign(t_u=users_relation_emb, t_i=items_relation_emb, 
                                     w_u=users_emb.detach(), w_i=items_emb.detach(), 
                                     o_space=o_space)
        uniform_relation = (self.lunif(users_relation_emb) + 
                            self.lunif(items_relation_emb)) / 2
        
        align_unbias = self.lalign(t_u=users_emb, t_i=items_emb, 
                                   w_u=users_relation_emb.detach(), w_i=items_relation_emb.detach())
        uniform_unbias = (self.lunif(x=users_emb) + 
                          self.lunif(x=items_emb)) / 2
                
        return align_relation, align_unbias, uniform_relation, uniform_unbias
    
    def computer(self):
        users_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight
        
        all_emb = torch.cat([users_emb, item_emb])
        embs = [all_emb]

        for layer in range(self.layers):
            all_emb = torch.sparse.mm(self.Graph_ui, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def lalign(self, t_u, t_i, w_u, w_i, o_space = False, alpha=2):
        if o_space:
            align_loss =  (t_u - t_i).norm(p=2, dim=1).pow(alpha)
            return align_loss.mean()

        w = torch.mul(w_u, w_i).sum(dim=1)
        w = torch.sigmoid(w)
        w = torch.clamp(w, min=0.1)
        
        align_loss = 1/w * (t_u - t_i).norm(p=2, dim=1).pow(alpha)
        return align_loss.mean()
        
    def lunif(self, x, t=2):
        mask = torch.triu(torch.ones(x.size(0), x.size(0), dtype=bool, device=x.device), diagonal=1)
        sq_pdist = torch.cdist(x,x)[mask]
        return sq_pdist.pow(2).mul(-t).exp().mean().log()
    
    def getSparseGraph(self):
        Graph_ui = None
        device = 'cuda'
        UserItemMat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        adj_mat_ui = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat_ui = adj_mat_ui.tolil()
        R_ui = UserItemMat.tolil()
        for data in self.train_data:
            user, item = data
            R_ui[user, item] = 1
        adj_mat_ui[:self.num_users, self.num_users:] = R_ui
        adj_mat_ui[self.num_users:, :self.num_users] = R_ui.T
        adj_mat_ui = adj_mat_ui.todok()

        norm_adj_ui = self.norm_adj_single(adj_mat_ui)
        Graph_ui = self._convert_sp_mat_to_sp_tensor(norm_adj_ui)
        Graph_ui = Graph_ui.coalesce().to(device)
        return Graph_ui

    def norm_adj_single(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        return norm_adj

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    def predict(self, user):
        user =  torch.from_numpy(np.array(user)).long().to('cuda')
        user_id = self.embedding_user_final[user]
        pred = torch.matmul(user_id, self.embedding_item_final.T)
        return pred
    
    def save_model(self, model, args):
        code_name = os.path.basename(__file__).split('.')[0]
        log_path = "model/{}/".format(code_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        torch.save({'model': model.state_dict()}, log_path + \
                   "%s_embed_size%d_reg%.5f_lr%0.5f_layer%.d_Gamma1_%.2f_Gamma2_%.2f.pth" % \
                          (args.dataset,
                           args.embedding_size, 
                           args.reg_rate, 
                           args.lr,  
                           args.layer,
                           args.gamma_1,
                           args.gamma_2))

def get_ui_pop(data_train, num_users, num_items):
    train_users = torch.IntTensor([i[0] for i in data_train])
    train_items = torch.IntTensor([i[1] for i in data_train])
    
    item_pop = torch.unique(train_items, return_counts=True)
    item_pop_id = item_pop[0].tolist()
    item_pop_cnt = item_pop[1]
    
    i_pop = torch.ones(num_items).long()
    i_pop[item_pop_id] = item_pop_cnt
    
    sorted_idx = torch.argsort(item_pop_cnt)
    unpop_threshold = int(len(sorted_idx)*0.8)-1   
    unpop_threshold_value = item_pop_cnt[sorted_idx[unpop_threshold]]
    
    min_value = unpop_threshold_value
    group_value = torch.zeros(2).long()
    for i in range(2):
        group_value[i] = min_value*(i+1)
    group_value[-1] = max(i_pop)
    
    for i in range(2):
        if i == 0:
            i_pop = torch.where((-1 < i_pop) & ( i_pop <= group_value[i]), i, i_pop)
        else:
            i_pop = torch.where((group_value[i-1] < i_pop) & ( i_pop <= group_value[i]), i, i_pop)
    
    
    user_pop = torch.unique(train_users, return_counts=True)
    user_pop_id = user_pop[0].tolist()
    
    user_pop_cnt = user_pop[1]
    u_pop = torch.zeros(num_users).long()
    u_pop[user_pop_id] = user_pop_cnt
    
    sorted_idx = torch.argsort(user_pop_cnt)
    unpop_threshold = int(len(sorted_idx)*0.8)-1
    unpop_threshold_value = user_pop_cnt[sorted_idx[unpop_threshold]]
    
    min_value = unpop_threshold_value
    group_value = torch.zeros(2).long()
    for i in range(2):
        group_value[i] = min_value*(i+1)
    group_value[-1] = max(u_pop)
    
    for i in range(2):
        if i == 0:
            u_pop = torch.where((-1 < u_pop) & ( u_pop <= group_value[i]), i, u_pop)
        else:
            u_pop = torch.where((group_value[i-1] < u_pop) & ( u_pop <= group_value[i]), i, u_pop)
    
    return u_pop, i_pop

def train(data_train, data_test, num_users, num_items):   
    device = torch.device('cuda')
    uctrl = uCtrl(data_train, num_users, num_items).to(device)
    
    train_dic, test_dic = get_ui_dict(data_train), get_ui_dict(data_test)
    user_list, item_list = sample_item(data_train, train_dic)
    
    train_datasets = uCtrl_train_dataset(user_list, item_list)
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, num_workers=2,shuffle=True)
    
    optimizer = optim.Adam(uctrl.parameters() , lr = args.lr, weight_decay = args.reg_rate)

    best_pre_5 = [0] * 12
    for epoch in range(args.epochs):
        uctrl.train()
        with tqdm(total=len(train_dataloader)) as t:
            for step, (user_list, item_list) in enumerate(train_dataloader):
                u = torch.from_numpy(np.array(user_list)).long().to(device)
                i = torch.from_numpy(np.array(item_list)).long().to(device)
                                
                align_relation, align_unbias, uniform_relation, uniform_unbias = uctrl(u, i)

                batch_loss = align_relation + align_unbias + args.gamma_1 * uniform_relation + args.gamma_2 * uniform_unbias
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                t.set_description(desc="Epoch {} train".format(epoch))
                t.update()
                
            t.set_postfix({'align_loss' : '{0:1.5f}'.format(align_relation),
                           'unif_loss' : '{0:1.5f}'.format(uniform_relation)})

        metric = []
        if (epoch+1) % args.verbose == 0:
            uctrl.eval()
            with torch.no_grad():
                with tqdm(total = num_users) as t_t:
                    for key in test_dic.keys():
                        len_train_list = len(train_dic[key])
                        rank_list = []
                        pred = uctrl.predict(key)
                        pred = pred.cpu().detach().numpy()
                        tmp = reckit.arg_top_k(pred,20+len_train_list)
                        for i in tmp:
                            if i in train_dic[key]:
                                continue
                            rank_list.append(i)
                        test_list = test_dic[key]
                        
                        p_3, r_3, ndcg_3 = precision_recall_ndcg_at_k(3, rank_list[:3], test_list)    
                        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, rank_list[:5], test_list)    
                        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, rank_list[:10], test_list)
                        p_20, r_20, ndcg_20 = precision_recall_ndcg_at_k(20, rank_list[:20], test_list)
            
                        metric.append([p_3, r_3, ndcg_3, p_5, r_5, ndcg_5, p_10, r_10, ndcg_10, p_20, r_20, ndcg_20])
                            
                        t_t.set_description(desc="Epoch {} test".format(epoch))
                        t_t.update()
                    
                metric = np.mean(np.array(metric),axis = 0).tolist()
                metric = [round(elem, 6) for elem in metric ]
                        
                print(metric)
                    
                logging.info("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                     .format(epoch, round(align_relation.item(),6), round(uniform_relation.item(),6), \
                        metric[0], metric[1], metric[2], \
                        metric[3], metric[4], metric[5], \
                        metric[6], metric[7], metric[8], \
                        metric[9], metric[10], metric[11]))
                 
                if metric[11] > best_pre_5[11]:
                    best_pre_5 = metric
                    uctrl.save_model(uctrl, args)
                    print("Find the best, save model.")
                
    print(best_pre_5)
    logging.info("Best:\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                 .format(best_pre_5[0], best_pre_5[1], best_pre_5[2], \
                         best_pre_5[3], best_pre_5[4], best_pre_5[5], \
                         best_pre_5[6], best_pre_5[7], best_pre_5[8], \
                         best_pre_5[9], best_pre_5[10], best_pre_5[11]))
        
    out_max(best_pre_5[0], best_pre_5[1], best_pre_5[2], 
            best_pre_5[3], best_pre_5[4], best_pre_5[5], 
            best_pre_5[6], best_pre_5[7], best_pre_5[8], 
            best_pre_5[9], best_pre_5[10], best_pre_5[11])

       
def out_max(pre3, rec3, ndcg3, pre5, rec5, ndcg5, pre10, rec10, ndcg10, pre20, rec20, ndcg20):
    code_name = os.path.basename(__file__).split('.')[0]
    log_path_ = "log/%s/" % (code_name)
    if not os.path.exists(log_path_):
        os.makedirs(log_path_)
    csv_path =  log_path_ + "%s.csv" % (args.dataset)
    log = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
            args.embedding_size, args.lr, args.reg_rate, 
            args.layer, args.gamma_1, args.gamma_2,
            pre3, rec3, ndcg3,
            pre5, rec5, ndcg5,
            pre10, rec10, ndcg10,
            pre20, rec20, ndcg20)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("embedding_size,learning_rate,reg_rate,layer,gamma_1,gamma_2,pre3,recall3,ndcg3,pre5,recall5,ndcg5,pre10,recall10,ndcg10,pre20,recall20,ndcg20" + '\n')
            f.write(log + '\n')  
            f.close()
    else:
        with open(csv_path, 'a+') as f:
            f.write(log + '\n')  
            f.close()

def get_ui_dict(data):
    res = {}
    for i in data:
        if i[0] not in res.keys():
            res[i[0]] = []
        res[i[0]].append(i[1])
    return res

def sample_item(data_train, train_dic):
    user_list = []
    item_list = []
    for entity in data_train:
        user_list.append(entity[0])
        item_list.append(entity[1])
        
    user_list = np.array(user_list)
    item_list = np.array(item_list)
    
    return user_list, item_list

def load_ui(path):
    num_users = -1
    num_items = -1
    data = []
    with open(path) as f:
        for line in f:
            line = [int(i) for i in line.split('\t')[:2]]
            data.append(line)
            num_users = max(line[0], num_users)
            num_items = max(line[1], num_items)
    num_users, num_items,  = num_users+1, num_items+1
    return data, num_users, num_items

def load_data(path):
    print('Loading train and test data...', end='')
    data_train, num_users, num_items = load_ui(path + '.train')
    data_test, num_users2, num_items2 = load_ui(path + '.test')
    num_users = max(num_users, num_users2)
    num_items = max(num_items, num_items2)
    print('Done.')
    print()
    print('Number of users: %d' % num_users)
    print('Number of items: %d' % num_items)
    print('Number of train data: %d' % len(data_train))
    print('Number of test data: %d' % len(data_test))

    logging.info('Number of users: %d' % num_users)
    logging.info('Number of items: %d' % num_items)
    logging.info('Number of train data: %d' % len(data_train))
    logging.info('Number of test data: %d' % len(data_test))
    
    return data_train, data_test, num_users, num_items

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN_Item")
    parser.add_argument('--dataset_path', nargs='?', default='./dataset/amazon-book/',
                        help='Data path.')      
    parser.add_argument('--dataset', nargs='?', default='amazon-book',
                        help='Name of the dataset.')  
    parser.add_argument('--batch_size', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--embedding_size', type=int,default=64,
                        help="the embedding size of lightGCN")   
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")   
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")    
    parser.add_argument('--reg_rate', type=float, default=0.0,
                        help='Regularization coefficient for user and item embeddings.')
    parser.add_argument('--gamma_1', type=float, default=2)
    parser.add_argument('--gamma_2', type=float, default=2)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Interval of evaluation.')
    return parser.parse_args()

if __name__ == '__main__':    
    code_name = os.path.basename(__file__).split('.')[0]
    args = parse_args()
    print(args)
    setup_seed(args.seed)
    log_path = "log/%s_%s/" % (code_name, strftime('%Y-%m-%d', localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = log_path + "%s_embed_size%.4f_reg%.5f_lr%0.5f_layer%.d_gamma1_%.2f_gamma2_%.2f_%s" % (
        args.dataset, args.embedding_size, args.reg_rate, args.lr, args.layer, args.gamma_1, args.gamma_2, strftime('%Y_%m_%d_%H', localtime()))
    logging.basicConfig(filename=log_path, level=logging.INFO)
    logging.info(args)

    data_train, data_test, num_users, num_items = load_data(args.dataset_path + args.dataset)
    train(data_train, data_test, num_users, num_items)








