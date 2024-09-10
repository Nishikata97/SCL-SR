import datetime
import pandas as pd
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import pickle


def clean_attribute(df_attribute, item_dict):
    df_attribute = df_attribute[df_attribute['item_id'].isin(item_dict.keys())]  
    df_attribute = df_attribute.drop_duplicates(subset=['item_id'], keep='first', ignore_index=True)
    print('No. items with attribute information:', df_attribute.shape[0])
    df_attribute.rename(columns={'item_id': 'true_item_id'}, inplace=True)

    item_id_list = []
    for index, row in tqdm(df_attribute.iterrows()):
        item_id_list.append(item_dict[row['true_item_id']])

    df_attribute['item_id'] = item_id_list
    return df_attribute

def item2attribure_csb(df_attribute):
    

    cat_id_array = df_attribute.cat_id.unique()
    cat_dict = {cid: i + 1 for i, cid in enumerate(cat_id_array)}
    df_attribute.cat_id = df_attribute.cat_id.map(cat_dict)

    seller_id_array = df_attribute.seller_id.unique()
    seller_dict = {sid: i + 1 for i, sid in enumerate(seller_id_array)}
    df_attribute.seller_id = df_attribute.seller_id.map(seller_dict)

    brand_id_array = df_attribute.brand_id.unique()
    brand_dict = {bid: i + 1 for i, bid in enumerate(brand_id_array)}
    df_attribute.brand_id = df_attribute.brand_id.map(brand_dict)

    item2cat = {}
    item2seller = {}
    item2brand = {}
    for index, row in tqdm(df_attribute.iterrows()):
        item2cat[row['item_id']] = row['cat_id']
        item2seller[row['item_id']] = row['seller_id']
        item2brand[row['item_id']] = row['brand_id']

    item2cat[0] = 0
    item2seller[0] = 0
    item2brand[0] = 0

    return cat_dict, seller_dict, brand_dict, item2cat, item2seller, item2brand


def item2attribute_cb(df_attribute):

    cat_id_array = df_attribute.cat_id.unique()
    cat_dict = {cid: i + 1 for i, cid in enumerate(cat_id_array)}
    df_attribute.cat_id = df_attribute.cat_id.map(cat_dict)

    brand_id_array = df_attribute.brand_id.unique()
    brand_dict = {bid: i + 1 for i, bid in enumerate(brand_id_array)}
    df_attribute.brand_id = df_attribute.brand_id.map(brand_dict)

    item2cat = {}
    item2brand = {}
    for index, row in tqdm(df_attribute.iterrows()):
        item2cat[row['item_id']] = row['cat_id']
        item2brand[row['item_id']] = row['brand_id']

    item2cat[0] = 0
    item2brand[0] = 0

    return cat_dict, brand_dict, item2cat, item2brand


def item2attribure_cs(df_attribute):
    cat_id_array = df_attribute.cat_id.unique()
    cat_dict = {cid: i + 1 for i, cid in enumerate(cat_id_array)}
    df_attribute.cat_id = df_attribute.cat_id.map(cat_dict)

    seller_id_array = df_attribute.seller_id.unique()
    seller_dict = {sid: i + 1 for i, sid in enumerate(seller_id_array)}
    df_attribute.seller_id = df_attribute.seller_id.map(seller_dict)

    item2cat = {}
    item2seller = {}

    for index, row in tqdm(df_attribute.iterrows()):
        item2cat[row['item_id']] = row['cat_id']
        item2seller[row['item_id']] = row['seller_id']

    item2cat[0] = 0
    item2seller[0] = 0

    return cat_dict, seller_dict, item2cat, item2seller

def load_tmall(raw_dataset_dir):
    user_log_file = raw_dataset_dir / 'data_format1/user_log_format1.csv'
    item_dict_file = str(pre_dataset_dir / 'item_dict.txt')
    df_attribute = pd.read_csv(user_log_file, sep=',')
    item_dict = pickle.load(open(item_dict_file, 'rb'))

    return df_attribute, item_dict

def save_attribute_cb(item2cat, item2brand):
    
    
    pickle.dump(item2cat, open(pre_dataset_dir / 'item2cat.txt', 'wb'))
    pickle.dump(item2brand, open(pre_dataset_dir / 'item2brand.txt', 'wb'))

def save_attribute_csb(item2cat, item2seller, item2brand):
    
    
    
    pickle.dump(item2cat, open(pre_dataset_dir / 'item2cat.txt', 'wb'))
    pickle.dump(item2seller, open(pre_dataset_dir / 'item2seller.txt', 'wb'))
    pickle.dump(item2brand, open(pre_dataset_dir / 'item2brand.txt', 'wb'))

def save_attribute_cs(item2cat, item2seller):
    
    
    pickle.dump(item2cat, open(pre_dataset_dir / 'item2cat.txt', 'wb'))
    pickle.dump(item2seller, open(pre_dataset_dir / 'item2seller.txt', 'wb'))

if __name__ == '__main__':
    np.random.seed(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='CIKM19',
        choices=['Tmall', 'Cosmetics', 'CIKM19', 'IJCAI16', 'Tianchi'],
        help='the dataset name',
    )
    parser.add_argument(
        '--raw-dataset-dir',
        default='/home/nishikata/practice/datasets/Multi-Behavior-datasets/{dataset}',
        help='the folder to save the raw dataset',
    )
    parser.add_argument(  
        '--pre-dataset-dir',
        default='../pre_datasets/{dataset}',
        help='the folder to save the preprocessed dataset',
    )
    args = parser.parse_args()
    print("-- Starting @ %ss" % datetime.datetime.now())
    print(f'Preprocess the dataset: {args.dataset}')

    raw_dataset_dir = Path(args.raw_dataset_dir.format(dataset=args.dataset))
    pre_dataset_dir = Path(args.pre_dataset_dir.format(dataset=args.dataset))
    pre_dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'Tmall':
        df_attribute, item_dict = load_tmall(raw_dataset_dir)
        df_attribute = clean_attribute(df_attribute, item_dict)
        df_attribute = df_attribute[['item_id', 'cat_id', 'seller_id', 'brand_id']]
        cat_dict, seller_dict, brand_dict, item2cat, item2seller, item2brand = item2attribure_csb(df_attribute)
        save_attribute_csb(item2cat, item2seller, item2brand)

    elif args.dataset == 'Cosmetics':
        subdatasets = ['2020-Feb.csv']
        
        df_cosmetics = pd.DataFrame(data=None,
                                    columns=['event_time', 'event_type', 'product_id', 'category_id', 'category_code',
                                             'brand', 'price', 'user_id', 'user_session'], )
        for dataset in subdatasets:
            dataset_file = raw_dataset_dir / dataset
            sub_df = pd.read_csv(dataset_file, index_col=None)
            df_cosmetics = pd.concat(objs=[df_cosmetics, sub_df], ignore_index=True)
        df_attribute = df_cosmetics[['event_time', 'event_type', 'product_id', 'category_id', 'brand', 'user_session']]
        df_attribute.columns = ['timestamp', 'action_type', 'item_id', 'cat_id', 'brand_id', 'session_id']
        item_dict_file = str(pre_dataset_dir / 'item_dict.txt')
        item_dict = pickle.load(open(item_dict_file, 'rb'))  

        df_attribute = clean_attribute(df_attribute, item_dict)
        df_attribute = df_attribute[['item_id', 'cat_id', 'brand_id']]
        cat_dict, brand_dict, item2cat, item2brand = item2attribute_cb(df_attribute)
        save_attribute_cb(item2cat, item2brand)
    elif args.dataset == 'CIKM19':
        item_file = raw_dataset_dir / 'item.csv'
        item_dict_file = str(pre_dataset_dir / 'item_dict.txt')
        df_attribute = pd.read_csv(item_file, sep=',')
        df_attribute.columns = ['item_id', 'cat_id', 'seller_id', 'brand_id']
        item_dict = pickle.load(open(item_dict_file, 'rb'))

        df_attribute = clean_attribute(df_attribute, item_dict)
        cat_dict, seller_dict, brand_dict, item2cat, item2seller, item2brand = item2attribure_csb(df_attribute)
        save_attribute_csb(item2cat, item2seller, item2brand)
    elif args.dataset == 'IJCAI16':
        item_file = raw_dataset_dir / 'ijcai2016_taobao.csv'
        item_dict_file = str(pre_dataset_dir / 'item_dict.txt')
        df_attribute = pd.read_csv(item_file, sep=',')
        df_attribute.columns = ['user_id', 'seller_id', 'item_id', 'cat_id', 'action_type', 'timestamp']
        df_attribute = df_attribute[['seller_id', 'item_id', 'cat_id']]
        item_dict = pickle.load(open(item_dict_file, 'rb'))

        df_attribute = clean_attribute(df_attribute, item_dict)
        cat_dict, seller_dict, item2cat, item2seller = item2attribure_cs(df_attribute)
        save_attribute_cs(item2cat, item2seller)
    elif args.dataset == 'Tianchi':
        item_file = raw_dataset_dir / 'tianchi_fresh_comp_train_user.csv'
        item_dict_file = str(pre_dataset_dir / 'item_dict.txt')
        df_attribute = pd.read_csv(item_file, sep=',')
        df_attribute.columns = ['user_id', 'item_id', 'action_type', 'user_geohash', 'cat_id', 'timestamp']
        df_attribute = df_attribute[['item_id', 'cat_id']]
        item_dict = pickle.load(open(item_dict_file, 'rb'))

        df_attribute = clean_attribute(df_attribute, item_dict)
        cat_id_array = df_attribute.cat_id.unique()
        cat_dict = {cid: i + 1 for i, cid in enumerate(cat_id_array)}
        df_attribute.cat_id = df_attribute.cat_id.map(cat_dict)
        item2cat = {}
        for index, row in tqdm(df_attribute.iterrows()):
            item2cat[row['item_id']] = row['cat_id']
        item2cat[0] = 0
        pickle.dump(item2cat, open(pre_dataset_dir / 'item2cat.txt', 'wb'))

    print('Done!')
