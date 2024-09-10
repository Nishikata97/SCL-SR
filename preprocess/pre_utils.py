import datetime
import argparse
from pathlib import Path
import pandas as pd
import pickle
import numpy as np
import csv
import operator

def get_session_id(df, interval):
    df_prev = df.shift()
    is_new_session = (df.user_id != df_prev.user_id) | (
        df['timestamp'] - df_prev['timestamp'] > interval
    )
    session_id = is_new_session.cumsum() - 1
    return session_id

def group_sessions(df, interval):
    session_id = get_session_id(df, interval)
    df = df.assign(session_id=session_id)
    return df

def filter_short_sessions(df, min_len=2):
    session_len = df.groupby('session_id', sort=False).size()
    long_sessions = session_len[session_len >= min_len].index
    df_long = df[df.session_id.isin(long_sessions)]
    return df_long

'''def filter_long_sessions(df, max_len=50):
    session_len = df.groupby('session_id', sort=False).size()
    short_sessions = session_len[session_len <= max_len].index
    df_short = df[df.session_id.isin(short_sessions)]
    return df_short'''

def truncate_long_sessions(df, max_len=50, is_sorted=False):
    if not is_sorted:
        df = df.sort_values(['session_id', 'timestamp'])
    item_idx = df.groupby('session_id').cumcount()
    df_t = df[item_idx <= max_len]
    return df_t

def filter_infreq_items(df, min_support=5):
    item_support = df.groupby('item_id', sort=False).size()
    freq_items = item_support[item_support >= min_support].index 
    df_freq = df[df.item_id.isin(freq_items)]
    return df_freq

def split_by_proportion(df, test_split=0.2):
    endtime = df.groupby('session_id', sort=False).timestamp.max()
    endtime = endtime.sort_values()
    num_tests = int(len(endtime) * test_split)
    test_session_ids = endtime.index[-num_tests:]
    df_train = df[~df.session_id.isin(test_session_ids)]
    df_test = df[df.session_id.isin(test_session_ids)]
    return df_train, df_test

'''def split_by_time(df, timedelta=100):
    max_time = df.timestamp.max() 
    end_time = df.groupby('session_id').timestamp.max() 
    split_time = max_time - timedelta 
    train_sids = end_time[end_time < split_time].index
    test_sids = end_time[end_time > split_time].index
    df_train = df[df.session_id.isin(train_sids)]
    df_test = df[df.session_id.isin(test_sids)]
    return df_train, df_test'''

def reorder_sessions_by_endtime(df):
    endtime = df.groupby('session_id', sort=False).timestamp.max()
    df_endtime = endtime.sort_values().reset_index()
    oid2nid = dict(zip(df_endtime.session_id, df_endtime.index))
    session_id_new = df.session_id.map(oid2nid)
    df = df.assign(session_id=session_id_new)
    df = df.sort_values(['session_id', 'timestamp'])
    return df

def handle_view_data(df, args):
    df = df.sort_values(['session_id', 'timestamp'])

    df = filter_short_sessions(df, min_len=args.min_len_session)
    df = filter_infreq_items(df, min_support=args.infreq_items)
    df = filter_short_sessions(df, min_len=args.min_len_session)
    
    df = truncate_long_sessions(df, max_len=args.max_len_session, is_sorted=True)

    splitting_interval = 0.2
    df_train, df_test = split_by_proportion(df, test_split=splitting_interval)
    print(f'Divide the dataset according to splitting interval= {splitting_interval}')
    print(f'Before filtering, No. of train_sessions: {df_train.session_id.nunique()}')
    print(f'Before filtering, No. of test_sessions: {df_test.session_id.nunique()}')

    return df_train, df_test

def handle_aux_tmall(df_train, df_test, df_aux, args):
    aux_end_time = df_aux.groupby('session_id').timestamp.max()
    split_time = df_train.timestamp.max()
    train_aux_sids = aux_end_time[aux_end_time < split_time].index
    df_aux = df_aux[df_aux.session_id.isin(train_aux_sids)]

    df_viewed_sessions = df_aux[df_aux['action_type'] == 'view']
    df_bought_sessions = df_aux[df_aux['action_type'] == 'purchase']
    df_viewed_sessions = df_viewed_sessions.drop_duplicates(subset=['item_id', 'session_id'], keep='first', inplace=False)
    df_bought_sessions = df_bought_sessions.drop_duplicates(subset=['item_id', 'session_id'], keep='first', inplace=False)
    df_viewed_sessions = filter_short_sessions(df_viewed_sessions, min_len=args.min_len_view)
    df_bought_sessions = filter_short_sessions(df_bought_sessions, min_len=args.min_len_buy)
    print('No. of items in viewed_session_df:', len(df_viewed_sessions['item_id'].unique()))
    print('No. of items in bought_session_df:', len(df_bought_sessions['item_id'].unique()))

    item_id_list = list(set(df_viewed_sessions['item_id'].unique()).intersection(set(df_bought_sessions['item_id'].unique())))  
    item_dict = {iid: i + 1 for i, iid in enumerate(item_id_list)}

    df_viewed_sessions = df_viewed_sessions[df_viewed_sessions['item_id'].isin(item_id_list)]
    df_viewed_sessions = filter_short_sessions(df_viewed_sessions, min_len=args.min_len_view)  
    df_bought_sessions = df_bought_sessions[df_bought_sessions['item_id'].isin(item_id_list)]
    df_bought_sessions = filter_short_sessions(df_bought_sessions, min_len=args.min_len_buy)  
    print('No. of items in viewed_session_df:', len(df_viewed_sessions['item_id'].unique()))
    print('No. of items in bought_session_df:', len(df_bought_sessions['item_id'].unique()))

    df_train = df_train[df_train['item_id'].isin(item_dict.keys())]
    df_train = filter_short_sessions(df_train, min_len=2)
    df_test = df_test[df_test['item_id'].isin(item_dict.keys())]
    df_test = filter_short_sessions(df_test, min_len=2)

    
    train_item_id_new = df_train.item_id.map(item_dict)
    df_train = df_train.assign(item_id=train_item_id_new)
    test_item_id_new = df_test.item_id.map(item_dict)
    df_test = df_test.assign(item_id=test_item_id_new)
    viewed_item_id_new = df_viewed_sessions['item_id'].map(item_dict)
    df_viewed_sessions = df_viewed_sessions.assign(item_id=viewed_item_id_new)
    bought_item_id_new = df_bought_sessions['item_id'].map(item_dict)
    df_bought_sessions = df_bought_sessions.assign(item_id=bought_item_id_new)

    return df_train, df_test, df_viewed_sessions, df_bought_sessions, item_dict

def handle_aux(df_train, df_test, df_aux, args):
    aux_end_time = df_aux.groupby('session_id').timestamp.max()
    split_time = df_train.timestamp.max()
    train_aux_sids = aux_end_time[aux_end_time < split_time].index
    df_aux = df_aux[df_aux.session_id.isin(train_aux_sids)]
    df_aux = df_aux[df_aux.item_id.isin(df_train.item_id.unique())]
    df_aux = filter_short_sessions(df_aux, min_len=args.min_len_buy)

    item_id_list = list(set(df_train.item_id.unique()).intersection(set(df_aux['item_id'].unique())))
    item_dict = {iid: i + 1 for i, iid in enumerate(item_id_list)}

    df_aux = df_aux[df_aux['item_id'].isin(item_id_list)]
    df_aux = filter_short_sessions(df_aux, min_len=args.min_len_buy)  
    print('No. of items in df_aux:', len(df_aux['item_id'].unique()))
    aux_item_id_new = df_aux['item_id'].map(item_dict)
    df_aux = df_aux.assign(item_id=aux_item_id_new)

    df_train = df_train[df_train['item_id'].isin(item_dict.keys())]
    df_train = filter_short_sessions(df_train, args.min_len_session)
    df_test = df_test[df_test['item_id'].isin(item_dict.keys())]
    df_test = filter_short_sessions(df_test, args.min_len_session)

    
    train_item_id_new = df_train.item_id.map(item_dict)
    df_train = df_train.assign(item_id=train_item_id_new)
    test_item_id_new = df_test.item_id.map(item_dict)
    df_test = df_test.assign(item_id=test_item_id_new)

    print(df_train.item_id.nunique())

    return df_train, df_test, df_aux, item_dict

def process_seqs(df):
    df = reorder_sessions_by_endtime(df)
    sessions = df.groupby('session_id').item_id.apply(lambda x: list(x)).reset_index()
    sessions.columns = ['session_id', 'session']
    all_seqs = sessions['session'].map(lambda x: x[:-1])

    out_seqs = []
    labs = []
    ids = []
    all_length = 0
    for index, row in sessions.iterrows():
        seq_id = row['session_id']
        seq = row['session']
        all_length += len(seq)
        for i in range(1, len(seq)):
            target_item = seq[-i]
            labs += [target_item]
            out_seqs += [seq[:-i]]
            ids += [seq_id]

    return out_seqs, labs, ids, all_length, all_seqs

def save_datasets(df_train, df_test, item_dict, pre_dataset_dir):
    tr_seqs, tr_labs, tr_ids, all_length_train, all_train_seqs = process_seqs(df_train)
    te_seqs, te_labs, te_ids, all_length_test, all_test_seqs = process_seqs(df_test)
    tra = (tr_ids, tr_seqs, tr_labs)
    tes = (te_ids, te_seqs, te_labs)
    print('No. of seqs in tr_seqs after process_seqs:', len(tr_seqs))  
    print('No. of seqs in te_seqs after process_seqs:', len(te_seqs))  

    print('avg length: ', (all_length_train + all_length_test) * 1.0 / (
            df_train['session_id'].nunique() + df_test['session_id'].nunique()))

    pre_dataset_dir.mkdir(parents=True, exist_ok=True)
    num_items = len(item_dict.keys())
    print('num_items: ', num_items)
    with open(pre_dataset_dir / 'num_items.txt', 'w') as f:
        f.write(str(num_items))

    pickle.dump(item_dict, open(pre_dataset_dir / 'item_dict.txt', 'wb'))

    pickle.dump(tra, open(pre_dataset_dir / 'train.txt', 'wb'))
    pickle.dump(tes, open(pre_dataset_dir / 'test.txt', 'wb'))
    pickle.dump(all_train_seqs, open(pre_dataset_dir / 'all_train_seq.txt', 'wb'))


def save_aux(df_view, df_bought, pre_dataset_dir):
    df_viewed_session = df_view
    df_viewed_session = reorder_sessions_by_endtime(df_viewed_session)
    viewed_sessions = df_viewed_session.groupby('session_id').item_id.apply(lambda x: ','.join(map(str, x)))
    viewed_sessions = viewed_sessions.apply(lambda x: list(map(int, x.split(',')))).values
    pickle.dump(viewed_sessions, open(pre_dataset_dir / 'viewed_sessions.txt', 'wb'))

    df_bought_session = df_bought
    df_bought_session = reorder_sessions_by_endtime(df_bought_session)
    bought_sessions = df_bought_session.groupby('session_id').item_id.apply(lambda x: ','.join(map(str, x)))
    bought_sessions = bought_sessions.apply(lambda x: list(map(int, x.split(',')))).values
    pickle.dump(bought_sessions, open(pre_dataset_dir / 'bought_sessions.txt', 'wb'))