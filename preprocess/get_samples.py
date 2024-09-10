import datetime
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
import pickle
def create_samples(sessions, num_items, num_sample):
    relation = []
    adj1 = [dict() for _ in range(num_items)]

    for s_i in range(len(sessions)):
        data = sessions[s_i]
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                relation.append([data[i], data[j]])
                relation.append([data[j], data[i]])

    for tup in tqdm(relation): 
        if tup[0] == 0:
            print('error')
        if tup[1] in adj1[tup[0]].keys():
            adj1[tup[0]][tup[1]] += 1
        else:
            adj1[tup[0]][tup[1]] = 1

    '''adj = [[] for _ in range(num_items)]
    weight = [[] for _ in range(num_items)]'''
    adj = {}
    weight = {}

    for t in range(num_items): 
        x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
        adj[t] = [v[0] for v in x]
        weight[t] = [v[1] for v in x]

    all_sample_num = 0
    for i in range(num_items):
        all_sample_num += len(adj[i])
    print(all_sample_num)
    print(all_sample_num / num_items)

    count = 0
    for i in range(num_items):
        if len(adj[i]) < num_sample:
            count += 1
        if len(adj[i]) < num_sample:
            adj[i] = adj[i] + [0 for _ in range(num_sample - len(adj[i]))]
            weight[i] = weight[i] + [0 for _ in range(num_sample - len(adj[i]))]
        adj[i] = adj[i][:num_sample]
        weight[i] = weight[i][:num_sample]

    print('No. items that less than num_sample:', count) 
    adj[0] = [0 for _ in range(num_sample)]
    weight[0] = [0 for _ in range(num_sample)]

    return adj, weight


def save_relations(co_viewed_items, co_viewed_weight, co_bought_items, co_bought_weight, pre_dataset_dir):
    pickle.dump(co_viewed_items, open(pre_dataset_dir / 'co_viewed_items.pkl', 'wb'))
    pickle.dump(co_viewed_weight, open(pre_dataset_dir / 'co_viewed_weight.pkl', 'wb'))
    pickle.dump(co_bought_items, open(pre_dataset_dir / 'co_bought_items.pkl', 'wb'))
    pickle.dump(co_bought_weight, open(pre_dataset_dir / 'co_bought_weight.pkl', 'wb'))


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
        '--pre-dataset-dir',
        default='../pre_datasets/{dataset}',
        help='the folder to save the preprocessed dataset',
    )
    args = parser.parse_args()
    print("-- Starting @ %ss" % datetime.datetime.now())
    print(f'Preprocess the dataset: {args.dataset}')

    pre_dataset_dir = Path(args.pre_dataset_dir.format(dataset=args.dataset))
    pre_dataset_dir.mkdir(parents=True, exist_ok=True)

    viewed_sessions = pickle.load(open(pre_dataset_dir / 'viewed_sessions.txt', 'rb'))
    bought_sessions = pickle.load(open(pre_dataset_dir / 'bought_sessions.txt', 'rb'))
    with open(pre_dataset_dir / 'num_items.txt', 'rb') as f:
        num_items = f.readlines()
    num_items = int(num_items[0]) + 1  
    if args.dataset == 'Tmall':
        num_sample = 4
    elif args.dataset == 'Cosmetics':
        num_sample = 6
    elif args.dataset == 'CIKM19':
        num_sample = 6
    elif args.dataset == 'IJCAI16':
        num_sample = 4
    elif args.dataset == 'Tianchi':
        num_sample = 4
    else:
        num_sample = 4

    co_viewed_items, co_viewed_weight = create_samples(viewed_sessions, num_items, num_sample)
    co_bought_items, co_bought_weight = create_samples(bought_sessions, num_items, num_sample)

    save_relations(co_viewed_items, co_viewed_weight, co_bought_items, co_bought_weight, pre_dataset_dir)

    print('Done!')
