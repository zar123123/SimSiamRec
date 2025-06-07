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
from tqdm import tqdm
from RankingMetrics import *

class GCLAU_train_dataset(Dataset):
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


class Predictor(nn.Module):
    def __init__(self,
                 hidden_size,
                 inner_size):
        super(Predictor, self).__init__()
        
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.act_fn = nn.functional.relu
        self.dense_2 = nn.Linear(inner_size, hidden_size)
    
    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        
        return hidden_states


class GCLAU(nn.Module):
    def __init__(self, data_train, data_test, num_users, num_items, embedding_size, pre_inner_size,
                 gamma, layer, lr, batch_size, eps, noise_type):
        super(GCLAU, self).__init__()
        self.train_data = data_train
        self.test_data = data_test
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.gamma = gamma
        self.layer = layer
        self.learning_rate = lr
        self.pre_inner_size = pre_inner_size
        self.batch_size = batch_size
        self.eps = eps
        self.noise_type = noise_type

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_size)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_size)
        self.embedding_user_final = None
        self.embedding_item_final = None
        
        self.Predictor = Predictor(self.embedding_size, self.pre_inner_size)

        nn.init.normal_(self.embedding_user.weight,0,0.01)
        nn.init.normal_(self.embedding_item.weight,0,0.01)
        
        #nn.init.xavier_uniform_(self.embedding_user.weight)
        #nn.init.xavier_uniform_(self.embedding_item.weight)
        
        self.Graph_ui = self.getSparseGraph()
        
    def generate_noise(self):
        if self.noise_type[0] == 'uniform_noise':
            self.noise_1 = torch.rand([self.num_users+self.num_items, self.embedding_size]).cuda()
        if self.noise_type[0] == 'Gaussian_noise':
            self.noise_1 = torch.randn(size=[self.num_users+self.num_items, self.embedding_size]).cuda()
        if self.noise_type[1] == 'uniform_noise':
            self.noise_2 = torch.rand([self.num_users+self.num_items, self.embedding_size]).cuda()
        if self.noise_type[1] == 'Gaussian_noise':
            self.noise_2 = torch.randn(size=[self.num_users+self.num_items, self.embedding_size]).cuda()
        
        self.noise_1 = nn.functional.normalize(self.noise_1)
        self.noise_2 = nn.functional.normalize(self.noise_2)

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
    
    def lalign(self, x, y, alpha=2):
        return (x - y).norm(p=2,dim=1).pow(alpha).mean()
    
    def lunif(self, x, t=2):
        mask = torch.triu(torch.ones(x.size(0), x.size(0), dtype=bool, device=x.device), diagonal=1)
        sq_pdist = torch.cdist(x,x)[mask]
        return sq_pdist.pow(2).mul(-t).exp().mean().log()
    
    def forward(self, users, items):
        all_users, all_items = self.computer(self.Graph_ui)
        users_emb = all_users[users]
        items_emb = all_items[items]
        
        users_emb = nn.functional.normalize(users_emb,dim=-1)
        items_emb = nn.functional.normalize(items_emb,dim=-1)
        
        align_loss = self.lalign(users_emb, items_emb) 
        unif_loss = (self.lunif(users_emb) + self.lunif(items_emb)) / 2
        
        all_users_1, all_items_1 = self.noise_computer(self.Graph_ui, 0)
        all_users_2, all_items_2 = self.noise_computer(self.Graph_ui, 1)
        
        user_emb_1 = all_users_1[users]
        user_emb_2 = all_users_2[users]
        item_emb_1 = all_items_1[items]
        item_emb_2 = all_items_2[items]
        
        pre_user_emb_1 = self.Predictor(user_emb_1)
        pre_user_emb_2 = self.Predictor(user_emb_2)
        
        pre_item_emb_1 = self.Predictor(item_emb_1)
        pre_item_emb_2 = self.Predictor(item_emb_2)
        '''
        pre_user_emb_1 = user_emb_1
        pre_user_emb_2 = user_emb_2
        
        pre_item_emb_1 = item_emb_1
        pre_item_emb_2 = item_emb_2
        '''
        user_emb_1 = nn.functional.normalize(user_emb_1, dim=1)
        user_emb_2 = nn.functional.normalize(user_emb_2, dim=1)
        
        item_emb_1 = nn.functional.normalize(item_emb_1, dim=1)
        item_emb_2 = nn.functional.normalize(item_emb_2, dim=1)
        
        pre_user_emb_1 = nn.functional.normalize(pre_user_emb_1, dim=1)
        pre_user_emb_2 = nn.functional.normalize(pre_user_emb_2, dim=1)
        
        pre_item_emb_1 = nn.functional.normalize(pre_item_emb_1, dim=1)
        pre_item_emb_2 = nn.functional.normalize(pre_item_emb_2, dim=1)
        
        loss_ssl_user = self.lalign(user_emb_1, pre_user_emb_2) + \
                        self.lalign(user_emb_2, pre_user_emb_1) 
                        
        loss_ssl_item = self.lalign(item_emb_1, pre_item_emb_2) + \
                        self.lalign(item_emb_2, pre_item_emb_1) 
                        
        self.embedding_user_final = all_users
        self.embedding_item_final = all_items
        
        return align_loss, unif_loss, loss_ssl_user + loss_ssl_item
    
    def predict(self, user):
        user_id = self.embedding_user_final[user]
        pred = torch.matmul(user_id, self.embedding_item_final.T)
        return pred
    
    def computer(self, Graph):
        users_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight

        all_emb = torch.cat([users_emb, item_emb])
        embs = []
        for layer in range(self.layer):
            all_emb = torch.sparse.mm(Graph, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def noise_computer(self, Graph, noise_type):
        users_emb = self.embedding_user.weight
        item_emb = self.embedding_item.weight
        random_noise = None
        all_emb = torch.cat([users_emb, item_emb])
        embs = []

        for layer in range(self.layer):
            all_emb = torch.sparse.mm(Graph, all_emb)
            if noise_type == 0:
                random_noise = self.noise_1
            if noise_type == 1:
                random_noise = self.noise_2
                
            all_emb += torch.mul(torch.sign(all_emb), random_noise) * self.eps
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def save_model(self, model, args):
        code_name = os.path.basename(__file__).split('.')[0]
        log_path = "model/{}/".format(code_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        torch.save({'model': model.state_dict()}, log_path + \
                   "%s_embed_size%d_reg%.5f_ssl_reg%.5f_lr%0.5f_eps%.3f_pre_inner_size%d_gamma%.3f_layer%d_noise_type%s_%s.pth" % \
                          (args.dataset,
                           args.embedding_size, 
                           args.reg_rate, 
                           args.ssl_reg, 
                           args.lr, 
                           args.eps, 
                           args.pre_inner_size, 
                           args.gamma, 
                           args.layer, 
                           args.noise_type[0], 
                           args.noise_type[1]))     

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

def get_ui_dict(data):
    res = {}
    for i in data:
        if i[0] not in res.keys():
            res[i[0]] = []
        res[i[0]].append(i[1])
    return res

def sample_item(data_train, train_dic, test_dic):
    user_list = []
    item_list = []
    for entity in data_train:
        user_list.append(entity[0])
        item_list.append(entity[1])
    
    max_inter_item = 0
    for key in train_dic.keys():
        if len(train_dic[key]) > max_inter_item:
            max_inter_item = len(train_dic[key])
    
    test_user_list = []
    test_item_list = []
    for key in test_dic.keys():
        test_user_list.append(key)
        test_item_list.append(test_dic[key])
        
    user_list = np.array(user_list)
    item_list = np.array(item_list)
    
    return user_list, item_list, test_user_list, test_item_list, max_inter_item



def train(train_data, test_data, num_users, num_items):
    device = torch.device('cuda')
    gclAU = GCLAU(data_train, data_test, num_users, num_items, args.embedding_size, args.pre_inner_size, \
                     args.gamma, args.layer, args.lr, args.batch_size, args.eps, args.noise_type).to(device)
    
    train_dic, test_dic = get_ui_dict(data_train), get_ui_dict(data_test) 
    user_list, item_list, test_user_list, test_item_list, max_inter_item = sample_item(data_train, train_dic, test_dic)
    
    train_datasets = GCLAU_train_dataset(user_list, item_list)
    train_dataloader = DataLoader(train_datasets, batch_size=args.batch_size, num_workers=2,shuffle=True)
    
    optimizer = optim.Adam(gclAU.parameters(), lr = args.lr, weight_decay=args.reg_rate)
    
    best_pre_5 = [0] * 12
    for epoch in range(args.epochs):
        gclAU.train()
        a = time.time()
        gclAU.generate_noise()
        with tqdm(total=len(train_dataloader)) as t:
            for step, (user_list, item_list) in enumerate(train_dataloader):
                u = torch.from_numpy(np.array(user_list)).long().to(device)
                i = torch.from_numpy(np.array(item_list)).long().to(device)
                                
                align_loss, unif_loss, ssl_loss = gclAU(u, i)

                batch_loss = align_loss + args.gamma * unif_loss + args.ssl_reg * ssl_loss
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                t.set_description(desc="Epoch {} train".format(epoch))
                t.update()
                
            t.set_postfix({'align_loss' : '{0:1.5f}'.format(align_loss),
                           'unif_loss' : '{0:1.5f}'.format(unif_loss),
                           'cl_loss' : '{0:1.5f}'.format(ssl_loss)
                           })
        b = time.time()
        print(b-a)
        
        metric = []
        if (epoch+1) % args.verbose == 0:
            gclAU.eval()
            with torch.no_grad():
                with tqdm(total = num_users) as t_t:
                    for key in test_dic.keys():
                        len_train_list = len(train_dic[key])
                        rank_list = []
                        pred = gclAU.predict(key)
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
                    
                logging.info("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                     .format(epoch, round(align_loss.item(),6), round(unif_loss.item(),6), round(ssl_loss.item(), 6), \
                        metric[0], metric[1], metric[2], \
                        metric[3], metric[4], metric[5], \
                        metric[6], metric[7], metric[8], \
                        metric[9], metric[10], metric[11]))
                 
                if metric[11] > best_pre_5[11]:
                    best_pre_5 = metric
                    gclAU.save_model(gclAU, args)
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
    log = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" % (
            args.embedding_size, args.lr, args.reg_rate, args.ssl_reg, 
            args.layer, args.pre_inner_size, args.gamma, args.eps,
            args.noise_type[0], args.noise_type[1],
            pre3, rec3, ndcg3,
            pre5, rec5, ndcg5,
            pre10, rec10, ndcg10,
            pre20, rec20, ndcg20)
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write("embedding_size,learning_rate,reg_rate,ssl_reg,layer,pre_inner_size,gamma,eps,noise_type,noise_type,pre3,recall3,ndcg3,pre5,recall5,ndcg5,pre10,recall10,ndcg10,pre20,recall20,ndcg20" + '\n')
            f.write(log + '\n')  
            f.close()
    else:
        with open(csv_path, 'a+') as f:
            f.write(log + '\n')  
            f.close()
    
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    
def parse_args():  
    parser = argparse.ArgumentParser(description="Go SimGCL")
    parser.add_argument('--dataset_path', nargs='?', default='./dataset/amazon-book/',
                        help='Data path.')      
    parser.add_argument('--dataset', nargs='?', default='amazon-book',
                        help='Name of the dataset.')  
    parser.add_argument('--batch_size', type=int,default=1024,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--embedding_size', type=int,default=64,
                        help="the embedding size of lightGCN")   
    parser.add_argument('--layer', type=int,default=1,
                        help="the layer num of lightGCN")   
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")   
    parser.add_argument('--reg_rate', type=float, default=0.0,
                        help='Regularization coefficient for user and item embeddings.')
    parser.add_argument('--ssl_reg', type=float, default=0.1,
                        help='Regularization coefficient for ssl.')
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--pre_inner_size', type=int, default=64)
    parser.add_argument('--noise_type', type=list, default=['uniform_noise','uniform_noise'], 
                        help='type of noise. Like [uniform_noise, uniform_noise],[uniform_noise, Gussian_noise],[Gussian_noise, Gussian_noise]')
    parser.add_argument('--epochs', type=int,default=50)
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
    log_path = log_path + "%s_embed_size%d_reg%.5f_ssl_reg%.5f_lr%0.5f_eps%.3f_pre_inner_size%d_gamma%.3f_layer%d_noise_type%s_%s_%s" % (
        args.dataset, args.embedding_size, args.reg_rate, args.ssl_reg, args.lr, args.eps, args.pre_inner_size, args.gamma, args.layer, 
        args.noise_type[0], args.noise_type[1], strftime('%Y_%m_%d_%H', localtime()))
    logging.basicConfig(filename=log_path, level=logging.INFO)     
    logging.info(args)

    data_train, data_test, num_users, num_items = load_data(args.dataset_path + args.dataset)
    train(data_train, data_test, num_users, num_items)

