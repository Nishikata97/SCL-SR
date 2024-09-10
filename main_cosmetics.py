import time
from pathlib import Path
from parse import parse_args
from utils.dataset import *
from utils.utils import *
from models.train_test import train_test
from models.model_cosmetics import *
import wandb


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


args = parse_args()


def main():
    init_seed(args.random_seed)
    
    
    args.num_coarse_sampling = 90
    args.beta = 0.02  
    args.gamma = 0  
    args.delta = 100  

    embeddings_dir = Path(args.embeddings_dir.format(dataset=args.dataset))
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir.format(dataset=args.dataset))
    models_dir.mkdir(parents=True, exist_ok=True)
    pre_dataset_dir = Path(args.pre_dataset_dir.format(dataset=args.dataset))
    train_sessions, test_sessions, all_train, num_items, co_viewed_items, co_bought_items = read_dataset(pre_dataset_dir, args)

    args.num_items = num_items + 1  
    item2cat, item2brand = load_cosmetics_attribute(pre_dataset_dir)
    args.num_cats = max(item2cat.values()) + 1 
    args.num_brands = max(item2brand.values()) + 1 
    print(args)
    wandb.init(name=time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + args.model, project=args.dataset)
    wandb.config.update(args)
    

    attribute_list = [item2cat, item2brand]
    relation_list = [co_viewed_items, co_bought_items]
    train_data = Data_cosmetics(train_sessions, all_train, args.num_items, attribute_list, relation_list)
    test_data = Data_cosmetics(test_sessions, all_train, args.num_items, attribute_list, relation_list)

    model = trans_to_cuda(Model_cosmetics(args, train_data.adjacency))
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    bad_counter = 0
    patience = 3

    for epoch in range(args.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_20, mrr_20, hit_10, mrr_10, hit_5, mrr_5 = train_test(args, model, train_data, test_data)

        embeddings_file = 'epoch-' + str(epoch) + '.pth'
        torch.save(model.embedding.weight.data, embeddings_dir / embeddings_file)
        model_file = 'epoch-' + str(epoch) + '_model.pth'
        torch.save(model.state_dict(), models_dir / model_file)

        flag = 0
        if hit_20 >= best_result[0]:
            best_result[0] = hit_20
            best_epoch[0] = epoch
            flag = 1
        if mrr_20 >= best_result[1]:
            best_result[1] = mrr_20
            best_epoch[1] = epoch
            flag = 1

        if hit_10 >= best_result[2]:
            best_result[2] = hit_10
            best_epoch[2] = epoch
            flag = 1
        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
            flag = 1

        if hit_5 >= best_result[4]:
            best_result[4] = hit_5
            best_epoch[4] = epoch
            flag = 1
        if mrr_5 >= best_result[5]:
            best_result[5] = mrr_5
            best_epoch[5] = epoch
            flag = 1
        print('Current Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit_20, mrr_20))
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f' % (hit_10, mrr_10))
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f' % (hit_5, mrr_5))

        wandb.log({"hit_20": hit_20, "mrr_20": mrr_20,
                   "hit_10": hit_10, "mrr_10": mrr_10,
                   "hit_5": hit_5, "mrr_5": mrr_5})

        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('\tRecall@10:\t%.4f\tMMR@10:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
        print('\tRecall@5:\t%.4f\tMMR@5:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[4], best_result[5], best_epoch[4], best_epoch[5]))
        bad_counter += 1 - flag
        if bad_counter >= patience:
            break

if __name__ == '__main__':
    main()
