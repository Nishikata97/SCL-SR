import numpy as np
from scipy.sparse import coo_matrix
import pickle
import torch
from torch.utils.data import Dataset
from operator import itemgetter
import random

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

def read_dataset(dataset_dir, args):
    train_data = pickle.load(open(dataset_dir / 'train.txt', 'rb'))
    if args.validation:
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(dataset_dir / 'test.txt', 'rb'))
    all_train_data = pickle.load(open(dataset_dir / 'all_train_seq.txt', 'rb'))

    with open(dataset_dir / 'item_dict.txt', 'rb') as f:
        item_dict = pickle.load(f)
        num_items = len(item_dict.keys()) 

    co_viewed_items = pickle.load(open(dataset_dir / 'co_viewed_items.pkl', 'rb'))
    co_bought_items = pickle.load(open(dataset_dir / 'co_bought_items.pkl', 'rb'))

    return train_data, test_data, all_train_data, num_items, co_viewed_items, co_bought_items


def load_tmall_attribute(dataset_dir):
    item2cat = pickle.load(open(dataset_dir / 'item2cat.txt', 'rb'))
    item2seller = pickle.load(open(dataset_dir / 'item2seller.txt', 'rb'))
    item2brand = pickle.load(open(dataset_dir / 'item2brand.txt', 'rb'))
    return item2cat, item2seller, item2brand

def load_cosmetics_attribute(dataset_dir):
    item2cat = pickle.load(open(dataset_dir / 'item2cat.txt', 'rb'))
    item2brand = pickle.load(open(dataset_dir / 'item2brand.txt', 'rb'))
    return item2cat, item2brand

def load_ijcai_attribute(dataset_dir):
    item2cat = pickle.load(open(dataset_dir / 'item2cat.txt', 'rb'))
    item2seller = pickle.load(open(dataset_dir / 'item2seller.txt', 'rb'))
    return item2cat, item2seller

def data_masks(all_sessions, n_node): 
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1: 
                break
            else:
                if sess[i]-1 not in adj.keys():
                    adj[sess[i]-1] = dict()
                    adj[sess[i]-1][sess[i]-1] = 1 
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i]-1][sess[i+1]-1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1 
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node-1, n_node-1)) 
    return coo

def data_masks_new(all_sessions, n_node): 
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            item_i = sess[i] - 1

            if i == len(sess)-1: 
                if item_i not in adj.keys(): 
                    adj[item_i] = dict()
                    adj[item_i][item_i] = 1  
                break
            else:
                item_j = sess[i + 1] - 1
                if item_i not in adj.keys(): 
                    adj[item_i] = dict()
                    adj[item_i][item_i] = 1  
                    adj[item_i][item_j] = 1  
                else:
                    if item_j not in adj[item_i].keys(): 
                        adj[item_i][item_j] = 1
                    else: 
                        adj[item_i][item_j] += 1
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i]
        for j in item.keys():
            row.append(i)
            col.append(j)
            data.append(adj[i][j])
    coo = coo_matrix((data, (row, col)), shape=(n_node-1, n_node-1)) 
    return coo

def handle_data(inputData, train_len=None):
    len_data = [len(nowData) for nowData in inputData]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    
    us_pois = [list(reversed(upois)) + [0] * (max_len - le) if le < max_len else list(reversed(upois[-max_len:]))
               for upois, le in zip(inputData, len_data)]
    us_msks = [[1] * le + [0] * (max_len - le) if le < max_len else [1] * max_len
               for le in len_data]
    return us_pois, us_msks, max_len

class Data(Dataset):
    def __init__(self, data, all_train, n_node, attribute_list, relation_list, train_len=None):
        inputs, mask, max_len = handle_data(data[1], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[2])
        self.mask = np.asarray(mask)
        self.length = len(data[1])
        self.max_len = max_len

        adj = data_masks_new(all_train, n_node) 
        
        self.adjacency = adj.multiply(1.0 / adj.sum(axis=0).reshape(1, -1))  

        
        item2cat, item2seller, item2brand = attribute_list
        co_viewed_items, co_bought_items, = relation_list
        self.cat_data = np.vectorize(item2cat.get)(self.inputs)
        self.seller_data = np.vectorize(item2seller.get)(self.inputs)
        self.brand_data = np.vectorize(item2brand.get)(self.inputs)
        self.viewed_dict = co_viewed_items
        self.bought_dict = co_bought_items

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        nonzero_elems = np.nonzero(u_input)[0]  
        session_len = [len(nonzero_elems)]

        
        cat = self.cat_data[index].tolist()
        seller = self.seller_data[index].tolist()
        brand = self.brand_data[index].tolist()
        viewed_items = [self.viewed_dict[item] for item in u_input]
        bought_items = [self.bought_dict[item] for item in u_input]


        return [torch.tensor(u_input), torch.tensor(target), torch.tensor(mask),
                torch.tensor(cat), torch.tensor(seller), torch.tensor(brand),
                torch.tensor(viewed_items), torch.tensor(bought_items)]

    def __len__(self):
        return self.length

class Data_cosmetics(Dataset):
    def __init__(self, data, all_train, n_node, attribute_list, relation_list, train_len=None):
        inputs, mask, max_len = handle_data(data[1], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[2])
        self.mask = np.asarray(mask)
        self.length = len(data[1])
        self.max_len = max_len

        adj = data_masks_new(all_train, n_node) 
        
        self.adjacency = adj.multiply(1.0 / adj.sum(axis=0).reshape(1, -1))  

        
        item2cat, item2brand = attribute_list
        co_viewed_items, co_bought_items, = relation_list
        self.cat_data = np.vectorize(item2cat.get)(self.inputs)
        self.brand_data = np.vectorize(item2brand.get)(self.inputs)
        self.viewed_dict = co_viewed_items
        self.bought_dict = co_bought_items

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        nonzero_elems = np.nonzero(u_input)[0]  
        session_len = [len(nonzero_elems)]

        
        cat = self.cat_data[index].tolist()
        brand = self.brand_data[index].tolist()
        viewed_items = [self.viewed_dict[item] for item in u_input]
        bought_items = [self.bought_dict[item] for item in u_input]

        return [torch.tensor(u_input), torch.tensor(target), torch.tensor(mask),
                torch.tensor(cat), torch.tensor(brand),
                torch.tensor(viewed_items), torch.tensor(bought_items)]

    def __len__(self):
        return self.length

class Data_ijcai(Dataset):
    def __init__(self, data, all_train, n_node, attribute_list, relation_list, train_len=None):
        inputs, mask, max_len = handle_data(data[1], train_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[2])
        self.mask = np.asarray(mask)
        self.length = len(data[1])
        self.max_len = max_len

        adj = data_masks_new(all_train, n_node) 
        
        self.adjacency = adj.multiply(1.0 / adj.sum(axis=0).reshape(1, -1))  

        
        item2cat, item2seller = attribute_list
        co_viewed_items, co_bought_items, = relation_list
        self.cat_data = np.vectorize(item2cat.get)(self.inputs)
        self.seller_data = np.vectorize(item2seller.get)(self.inputs)
        self.viewed_dict = co_viewed_items
        self.bought_dict = co_bought_items

    def __getitem__(self, index):
        u_input, mask, target = self.inputs[index], self.mask[index], self.targets[index]

        nonzero_elems = np.nonzero(u_input)[0]  
        session_len = [len(nonzero_elems)]

        
        cat = self.cat_data[index].tolist()
        seller = self.seller_data[index].tolist()
        viewed_items = [self.viewed_dict[item] for item in u_input]
        bought_items = [self.bought_dict[item] for item in u_input]

        return [torch.tensor(u_input), torch.tensor(target), torch.tensor(mask),
                torch.tensor(cat), torch.tensor(seller),
                torch.tensor(viewed_items), torch.tensor(bought_items)]

    def __len__(self):
        return self.length