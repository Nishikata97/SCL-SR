import datetime
import argparse
from pathlib import Path
import pandas as pd
import pickle
from pre_utils import *


def load_tmall(raw_dataset_dir):
    data_file = raw_dataset_dir / 'dataset15.csv'
    with open(data_file, 'r') as tmall_file:
        header = tmall_file.readline()
        tmall_data = []
        for line in tmall_file:
            data = line[:-1].split('\t')
            user_id = int(data[0])
            item_id = int(data[1])
            session_id = int(data[2])
            time_stamp = int(float(data[3]))
            
            if int(data[2]) > 120000:  
                break
            tmall_data.append([user_id, item_id, session_id, time_stamp])
    df_tmall = pd.DataFrame(tmall_data, columns=['userId', 'item_id', 'session_id', 'timestamp'])

    user_log_file = raw_dataset_dir / 'data_format1/user_log_format1.csv'
    df_aux = pd.read_csv(user_log_file, sep=',')
    df_aux = df_aux[['user_id', 'item_id', 'time_stamp', 'action_type']]
    df_aux.columns = ['user_id', 'item_id', 'timestamp', 'action_type']
    df_aux['session_id'] = df_aux['user_id'].astype(str) + '-' + df_aux['timestamp'].astype(str) 

    return df_tmall, df_aux

def load_cosmetics(raw_dataset_dir):
    subdatasets = ['2020-Feb.csv']
    
    df_cosmetics = pd.DataFrame(data=None,
                                columns=['event_time', 'event_type', 'product_id', 'category_id', 'category_code',
                                         'brand', 'price', 'user_id', 'user_session'], )
    for dataset in subdatasets:
        dataset_file = raw_dataset_dir / dataset
        df_sub = pd.read_csv(dataset_file, index_col=None)
        df_cosmetics = pd.concat(objs=[df_cosmetics, df_sub], ignore_index=True)
    df_cosmetics = df_cosmetics[['event_time', 'event_type', 'product_id', 'user_session']]
    df_cosmetics.columns = ['timestamp', 'action_type', 'item_id', 'session_id']
    df_cosmetics['timestamp'] = pd.to_datetime(df_cosmetics['timestamp'], format="%Y-%m-%d %H:%M:%S UTC",
                                               utc=True).map(pd.Timestamp.timestamp)
    df_cosmetics['timestamp'] = df_cosmetics['timestamp'].astype(int)
    df = df_cosmetics[df_cosmetics['action_type'] == 'view']
    df_aux = df_cosmetics[df_cosmetics['action_type'] == 'purchase']
    df = df.sort_values(['session_id', 'timestamp'])
    df_aux = df_aux.sort_values(['session_id', 'timestamp'])
    del df_cosmetics

    return df, df_aux

def load_cikm(raw_dataset_dir):
    datasets = 'user_behavior.csv'

    dataset_file = raw_dataset_dir / datasets
    df_cikm = pd.read_csv(dataset_file, index_col=None)
    df_cikm.columns = ['user_id', 'item_id', 'action_type', 'timestamp']
    df_cikm = df_cikm.sort_values(['user_id', 'timestamp'])
    
    view_session_interval = 3600  
    buy_session_interval = 86400 
    df = df_cikm[df_cikm['action_type'] == 'pv']
    df_aux = df_cikm[df_cikm['action_type'] == 'buy']

    del df_cikm

    df = group_sessions(df, view_session_interval)
    df = df.sort_values(['session_id', 'timestamp'])
    print('No. of sessions:', df['session_id'].max())
    df = df[df['session_id'] < 2000000]
    df_aux = group_sessions(df_aux, buy_session_interval)
    df_aux = df_aux.sort_values(['session_id', 'timestamp'])

    return df, df_aux


def load_fliggy(raw_dataset_dir):
    dataset_file = raw_dataset_dir / 'user_item_behavior_history.csv'
    df_fliggy = pd.read_csv(dataset_file, index_col=None)

    return df_fliggy, df_aux

def load_jdata2018(raw_dataset_dir):
    datasets = 'jdata_action.csv'
    attributes = 'jdata_product.csv'

    dataset_file = raw_dataset_dir / datasets
    attribute_file = raw_dataset_dir / attributes
    df_jdata18 = pd.read_csv(dataset_file, index_col=None)
    df_jdata18.columns = ['user_id', 'item_id', 'timestamp', 'session_id', 'action_type']
    df_jdata18 = df_jdata18[df_jdata18['timestamp'] >= '2018-04-01 00:00:00']
    df_jdata18 = df_jdata18.sort_values(['user_id', 'timestamp'])
    df_jdata18 = df_jdata18.sort_values(['session_id', 'timestamp'])
    df = df_jdata18[df_jdata18['action_type'] == 1]
    df_aux = df_jdata18[df_jdata18['action_type'] == 2]
    del df_jdata18

    attribute_df = pd.read_csv(attribute_file, index_col=None)
    attribute_df.columns = ['item_id', 'brand_id', 'seller_id', 'cat_id', 'market_time']
    attribute_df = attribute_df.drop_duplicates(subset=['item_id'], keep='first', ignore_index=True)
    df = df[df['item_id'].isin(attribute_df['item_id'].unique())]

    return df, df_aux

def load_ijcai16(raw_dataset_dir):
    datasets = 'ijcai2016_taobao.csv'
    dataset_file = raw_dataset_dir / datasets

    df_ijcai2016 = pd.read_csv(dataset_file, index_col=None)
    df_ijcai2016.columns = ['user_id', 'seller_id', 'item_id', 'cat_id', 'action_type', 'timestamp']
    df_ijcai2016 = df_ijcai2016.sort_values(['user_id', 'timestamp'])

    view_session_interval = 1  
    buy_session_interval = 3
    df = df_ijcai2016[df_ijcai2016['action_type'] == 0]
    df_aux = df_ijcai2016[df_ijcai2016['action_type'] == 1]
    df = group_sessions(df, view_session_interval)
    df = df.sort_values(['session_id', 'timestamp'])
    df = df[df['session_id'] <= 120000]

    df_aux = group_sessions(df_aux, buy_session_interval)
    df_aux = df_aux.sort_values(['session_id', 'timestamp'])

    df = df.drop_duplicates(subset=['item_id', 'session_id'], keep='first', inplace=False) 
    df_aux = df_aux.drop_duplicates(subset=['item_id', 'session_id'], keep='first', inplace=False) 
    del df_ijcai2016

    return df, df_aux


def load_tianchi(raw_dataset_dir):
    
    datasets = 'tianchi_fresh_comp_train_user.csv'
    dataset_file = raw_dataset_dir / datasets
    df_tianchi = pd.read_csv(dataset_file, index_col=None)
    df_tianchi.columns = ['user_id', 'item_id', 'action_type', 'user_geohash', 'cat_id', 'timestamp']
    df_tianchi['timestamp'] = pd.to_datetime(df_tianchi['timestamp'], format="%Y-%m-%d %H").map(pd.Timestamp.timestamp)
    df_tianchi['timestamp'] = df_tianchi['timestamp'].astype(int)
    df_tianchi = df_tianchi.sort_values(['user_id', 'timestamp'])

    view_session_interval = 3600  
    buy_session_interval = 86400  
    df = df_tianchi[df_tianchi['action_type'] == 1]
    df_aux = df_tianchi[df_tianchi['action_type'] == 4]
    df = group_sessions(df, view_session_interval)
    df = df.sort_values(['session_id', 'timestamp'])
    df_aux = group_sessions(df_aux, buy_session_interval)
    df_aux = df_aux.sort_values(['session_id', 'timestamp'])

    return df, df_aux


def preprocess(df, df_aux, args, pre_dataset_dir):
    print(args)

    df = df.sort_values(['session_id', 'timestamp'])
    df_aux = df_aux.sort_values(['session_id', 'timestamp'])
    df_train, df_test = handle_view_data(df, args)
    df_aux = df_aux[df_aux.item_id.isin(df_train.item_id.unique())]

    id2action_tmall = {0: 'view', 2: 'purchase'}
    if args.dataset == 'Tmall':
        df_aux['action_type'] = df_aux['action_type'].map(id2action_tmall)
        df_train, df_test, df_viewed_sessions, df_bought_sessions, item_dict = handle_aux_tmall(df_train, df_test, df_aux, args)
        train_gru = df_train[['session_id', 'item_id', 'timestamp']]
        test_gru = df_test[['session_id', 'item_id', 'timestamp']]
        pickle.dump(train_gru, open(pre_dataset_dir / 'train_gru.txt', 'wb'))
        pickle.dump(test_gru, open(pre_dataset_dir / 'test_gru.txt', 'wb'))
        save_datasets(df_train, df_test, item_dict, pre_dataset_dir)
        save_aux(df_viewed_sessions, df_bought_sessions, pre_dataset_dir)
    else:
        df_train, df_test, df_aux, item_dict = handle_aux(df_train, df_test, df_aux, args)
        train_gru = df_train[['session_id', 'item_id', 'timestamp']]
        test_gru = df_test[['session_id', 'item_id', 'timestamp']]
        pickle.dump(train_gru, open(pre_dataset_dir / 'train_gru.txt', 'wb'))
        pickle.dump(test_gru, open(pre_dataset_dir / 'test_gru.txt', 'wb'))
        save_datasets(df_train, df_test, item_dict, pre_dataset_dir)
        save_aux(df_train, df_aux, pre_dataset_dir)

    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='Tmall',
        choices=['Tmall', 'Cosmetics', 'CIKM19', 'JData2018', 'IJCAI16', 'Tianchi'],
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

    parser.add_argument('--min-len-session', type=int, default=2, help='Min length of bought_sessions')
    parser.add_argument('--max-len-session', type=int, default=50, help='Min length of bought_sessions')
    parser.add_argument('--infreq-items', type=int, default=5, help='Min length of viewed_sessions')
    parser.add_argument('--min-len-view', type=int, default=5, help='Min length of viewed_sessions')
    parser.add_argument('--min-len-buy', type=int, default=5, help='Min length of bought_sessions')
    args = parser.parse_args()
    print(f'Preprocess the dataset: {args.dataset}')

    amazon_datasets = ['Electronics', 'Grocery_and_Gourmet_Food', 'Cell_Phones_and_Accessories', 'Home_and_Kitchen']

    raw_dataset_dir = Path(args.raw_dataset_dir.format(dataset=args.dataset))
    pre_dataset_dir = Path(args.pre_dataset_dir.format(dataset=args.dataset))
    pre_dataset_dir.mkdir(parents=True, exist_ok=True)

    print("-- Starting @ %ss" % datetime.datetime.now())


    if args.dataset == 'Tmall':
        df, df_aux = load_tmall(raw_dataset_dir)
        args.min_len_session = 2
        args.infreq_items = 5
        args.max_len_session = 50

        args.min_len_view = 5
        args.min_len_buy = 5
        preprocess(df, df_aux, args, pre_dataset_dir)
    elif args.dataset == 'Cosmetics':
        df, df_aux = load_cosmetics(raw_dataset_dir)
        args.min_len_session = 2
        args.infreq_items = 5
        args.max_len_session = 50

        args.min_len_buy = 10
        preprocess(df, df_aux, args, pre_dataset_dir)
    elif args.dataset == 'CIKM19':
        df, df_aux = load_cikm(raw_dataset_dir)
        args.min_len_session = 2
        args.infreq_items = 5
        args.max_len_session = 50

        args.min_len_buy = 10
        preprocess(df, df_aux, args, pre_dataset_dir)
    elif args.dataset == 'Fliggy':
        df, df_aux = load_fliggy(raw_dataset_dir)
    elif args.dataset == 'JData2018':
        df, df_aux = load_jdata2018(raw_dataset_dir)
        args.min_len_session = 2
        args.infreq_items = 5
        args.max_len_session = 50

        args.min_len_view = 5
        args.min_len_buy = 5
        preprocess(df, df_aux, args, pre_dataset_dir)
    elif args.dataset == 'IJCAI16':
        df, df_aux = load_ijcai16(raw_dataset_dir) 
        args.min_len_session = 2
        args.infreq_items = 5
        args.max_len_session = 50

        args.min_len_view = 5
        args.min_len_buy = 5
        preprocess(df, df_aux, args, pre_dataset_dir)
    elif args.dataset == 'Tianchi':
        df, df_aux = load_tianchi(raw_dataset_dir)  
        args.min_len_session = 2
        args.infreq_items = 5
        args.max_len_session = 50

        args.min_len_view = 5
        args.min_len_buy = 5
        preprocess(df, df_aux, args, pre_dataset_dir)
    else:
        df = pd.DataFrame(data=None, columns=['event_time', 'event_type', 'product_id', 'category_id', 'category_code', 'brand', 'price', 'user_id', 'user_session'])
