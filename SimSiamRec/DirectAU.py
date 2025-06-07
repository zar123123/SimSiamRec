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
import time 
from time import strftime
from time import localtime
from torch.utils.data import DataLoader, Dataset
import argparse
import reckit
import sys
from tqdm import tqdm
from RankingMetrics import *


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class DirectAU_train_dataset(Dataset):
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


class DirectAU(nn.Module):
    def __init__(self, num_user, num_item):
        super(DirectAU, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.embedding_size = args.embedding_size

        self.embedding_user = nn.Embedding(num_user, self.embedding_size)
        self.embedding_item = nn.Embedding(num_item, self.embedding_size)
        self.present_train_batch=0
        
        nn.init.normal_(self.embedding_user.weight, 0, 0.01)
        nn.init.normal_(self.embedding_item.weight, 0, 0.01)
    
    def lalign(self, x, y, alpha=2):
        return (x - y).norm(p=2,dim=1).pow(alpha).mean()
    
    def lunif(self, x, t=2):
        #sq_pdist = torch.pdist(x, p=2).pow(2)
        #return sq_pdist.mul(-t).exp().mean().log()
        
        mask = torch.triu(torch.ones(x.size(0), x.size(0), dtype=bool, device=x.device), diagonal=1)
        sq_pdist = torch.cdist(x,x)[mask]
        return sq_pdist.pow(2).mul(-t).exp().mean().log()
        
        
    def forward(self, users, items):        
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        
        users_emb = nn.functional.normalize(users_emb,dim=-1)
        items_emb = nn.functional.normalize(items_emb,dim=-1)
        
        align_loss = self.lalign(users_emb, items_emb)
        unif_loss = (self.lunif(users_emb) + self.lunif(items_emb)) / 2
        return align_loss, unif_loss
    
    def predict(self,user):
        user =  torch.from_numpy(np.array(user)).long().to('cuda')
        users_emb = self.embedding_user(user)
        item_emb = self.embedding_item.weight.T
        pred = torch.matmul(users_emb, item_emb)
        return pred 
    
    def save_model(self, model, args):
        code_name = os.path.basename(__file__).split('.')[0]
        log_path = "model/{}/".format(code_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        torch.save({'model': model.state_dict()}, log_path + \
                   "%s_embed_size%d_reg%.5f_lr%0.5f_gamma%.3f.pth" % \
                          (args.dataset,
                           args.embedding_size, 
                           args.reg_rate, 
                           args.learning_rate,  
                           args.gamma))
      
def train(num_users, num_items, data_train, data_test):   
    device = torch.device('cuda')
    directau = DirectAU(num_users, num_items).to(device)
    
    train_dic, test_dic = get_ui_dict(data_train), get_ui_dict(data_test)
    user_list, item_list = sample_item(data_train, train_dic)
    
    train_datasets = DirectAU_train_dataset(user_list, item_list)
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, num_workers=2,shuffle=True)
    
    optimizer = optim.Adam(directau.parameters() , lr = args.learning_rate, weight_decay = args.reg_rate)
    
    best_pre_5 = [0] * 12
    for epoch in range(args.epochs):
        directau.train()
        a = time.time()
        with tqdm(total=len(train_dataloader)) as t:
            for step, (user_list, item_list) in enumerate(train_dataloader):
                u = torch.from_numpy(np.array(user_list)).long().to(device)
                i = torch.from_numpy(np.array(item_list)).long().to(device)
                                
                align_loss, unif_loss = directau(u, i)

                batch_loss = align_loss + args.gamma * unif_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                t.set_description(desc="Epoch {} train".format(epoch))
                t.update()
                
            t.set_postfix({'align_loss' : '{0:1.5f}'.format(align_loss),
                           'unif_loss' : '{0:1.5f}'.format(unif_loss)})
        b = time.time()
        print(b-a)
        
        metric = []
        if (epoch+1) % args.verbose == 0:
            directau.eval()
            with torch.no_grad():
                with tqdm(total = num_users) as t_t:
                    for key in test_dic.keys():
                        len_train_list = len(train_dic[key])
                        rank_list = []
                        pred = directau.predict(key)
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
                     .format(epoch, round(align_loss.item(),6), round(unif_loss.item(),6), \
                        metric[0], metric[1], metric[2], \
                        metric[3], metric[4], metric[5], \
                        metric[6], metric[7], metric[8], \
                        metric[9], metric[10], metric[11]))
                 
                if metric[11] > best_pre_5[11]:
                    best_pre_5 = metric
                    directau.save_model(directau, args)
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
    log = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
            args.embedding_size, args.learning_rate, args.reg_rate, args.gamma, 
            pre3, rec3, ndcg3,
            pre5, rec5, ndcg5,
            pre10, rec10, ndcg10,
            pre20, rec20, ndcg20)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("embedding_size,learning_rate,reg_rate,gamma,pre3,recall3,ndcg3,pre5,recall5,ndcg5,pre10,recall10,ndcg10,pre20,recall20,ndcg20" + '\n')
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
    num_users, num_items = num_users+1, num_items+1
    return data, num_users, num_items 

def load_data(path):
    print('Loading train and test data...', end='')
    sys.stdout.flush()
    train_data, num_users, num_items = load_ui(path+'.train')
    test_data, num_users2, num_items2 = load_ui(path+'.test')
    num_users = max(num_users, num_users2)
    num_items = max(num_items, num_items2)

    print('Number of users: %d'%num_users)
    print('Number of items: %d'%num_items)
    print('Number of train data: %d'%len(train_data))
    print('Number of test data: %d'%len(test_data))
    logging.info('Number of users: %d'%num_users)
    logging.info('Number of items: %d'%num_items)
    logging.info('Number of train data: %d'%len(train_data))
    logging.info('Number of test data: %d'%len(test_data))
    sys.stdout.flush()
    return train_data, test_data, num_users, num_items

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)  
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='DeepRec')
    parser.add_argument('--model', choices=['CDAE', 'CML', 'NeuMF', 'GMF', 'MLP', 'BPRMF', 'JRL', 'LRML'],
                        default='BPRMF') 
    parser.add_argument('--dataset_path', nargs='?', default='./dataset/douban_book/',
                        help='Data path.')
    parser.add_argument('--dataset', nargs='?', default='douban_book',
                        help='Name of the dataset.')     
    parser.add_argument('--learning_rate', type=float, default=1e-3) 
    parser.add_argument('--reg_rate', type=float, default=0.0001)
    parser.add_argument('--embedding_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024) 
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    return parser.parse_args()

if __name__ == '__main__':
    setup_seed(100)
    args = parse_args()
    print(args)    
    code_name = os.path.basename(__file__).split('.')[0]
    log_path = "log/%s_%s/" % (code_name,strftime('%Y-%m-%d', localtime()))
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_path = log_path + "%s_embed_size%.4f_reg%.5f_gamma%.5f_lr%0.5f_%s" % (
        args.dataset, args.embedding_size, args.reg_rate, args.gamma, args.learning_rate, strftime('%Y_%m_%d_%H', localtime()))
    logging.basicConfig(filename=log_path,
                        level=logging.INFO)  
    logging.info(args) 

    train_data, test_data, num_users, num_items = load_data(args.dataset_path + args.dataset)
    
    train(num_users, num_items, train_data, test_data)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        