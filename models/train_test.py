import datetime
from tqdm import tqdm
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.sparse
import wandb
from utils.utils import trans_to_cuda, trans_to_cpu


def forward_tmall(model, data):
    inputs, targets, masks, cats, sellers, brands, viewed_items, bought_items = data

    inputs = trans_to_cuda(inputs).long()
    targets = trans_to_cuda(targets.long())
    masks = trans_to_cuda(masks).long()

    cats = trans_to_cuda(cats).long()
    sellers = trans_to_cuda(sellers).long()
    brands = trans_to_cuda(brands).long()
    attributes = [cats, sellers, brands]

    viewed_items = trans_to_cuda(viewed_items).long()
    bought_items = trans_to_cuda(bought_items).long()

    scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess \
        = model(inputs, targets, masks, attributes, viewed_items, bought_items)
    return targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess


def forward_tianchi(model, data):
    inputs, targets, masks, cats, viewed_items, bought_items = data

    inputs = trans_to_cuda(inputs).long()
    targets = trans_to_cuda(targets.long())
    masks = trans_to_cuda(masks).long()

    cats = trans_to_cuda(cats).long()
    attributes = cats

    viewed_items = trans_to_cuda(viewed_items).long()
    bought_items = trans_to_cuda(bought_items).long()

    scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess \
        = model(inputs, targets, masks, attributes, viewed_items, bought_items)
    return targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess

def forward_2attributes(model, data):
    inputs, targets, masks, attribute_1, attribute_2, viewed_items, bought_items = data

    inputs = trans_to_cuda(inputs).long()
    targets = trans_to_cuda(targets).long()
    masks = trans_to_cuda(masks).long()

    attribute_1 = trans_to_cuda(attribute_1).long()
    attribute_2 = trans_to_cuda(attribute_2).long()
    attributes = [attribute_1, attribute_2]

    viewed_items = trans_to_cuda(viewed_items).long()
    bought_items = trans_to_cuda(bought_items).long()

    scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = model(inputs, targets, masks, attributes, viewed_items, bought_items)
    return targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess

def train_test(args, model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for batch_i, tra_data in tqdm(enumerate(train_loader), total=len(train_loader)):
        model.optimizer.zero_grad()
        dataset_3attributes = ['Tmall', 'CIKM19']
        if args.dataset in dataset_3attributes:
            targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = forward_tmall(model, tra_data)
        elif args.dataset == 'Tianchi':
            targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = forward_tianchi(model, tra_data)
        else:
            targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = forward_2attributes(model, tra_data)
        loss = loss_rec + args.beta * loss_contra + args.gamma * loss_batch_sess + args.delta * loss_attribute + loss_reconstruction
        loss.backward()
        model.optimizer.step()
        total_loss += loss

        wandb.log({"total_loss": loss})
        wandb.log({"loss_rec": loss_rec})
        wandb.log({"loss_contra": loss_contra})
        wandb.log({"loss_batch_sess": loss_batch_sess})
        wandb.log({"loss_attribute": loss_attribute})
        wandb.log({"loss_reconstruction": loss_reconstruction})

    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)

    result = []
    hit_20, mrr_20 = [], []
    hit_10, mrr_10 = [], []
    hit_5, mrr_5 = [], []

    dataset_3attributes = ['Tmall', 'CIKM19']
    for tes_data in test_loader:
        if args.dataset in dataset_3attributes:
            targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = forward_tmall(model, tes_data)
        elif args.dataset == 'Tianchi':
            targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = forward_tianchi(model, tes_data)
        else:
            targets, scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess = forward_2attributes(model, tes_data)

        tes_loss = loss_rec + args.beta * loss_contra + args.gamma * loss_batch_sess + args.delta * loss_attribute + loss_reconstruction
        wandb.log({"tes_loss": tes_loss})
        sub_20_scores = scores_item.topk(20)[1]  
        sub_20_scores = trans_to_cpu(sub_20_scores).detach().numpy()
        targets = targets.cpu().numpy()  
        for score, target in zip(sub_20_scores, targets):
            hit_20.append(np.isin(target - 1, score))  
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 / (np.where(score == target - 1)[0][0] + 1))  

        sub_10_scores = scores_item.topk(10)[1]
        sub_10_scores = trans_to_cpu(sub_10_scores).detach().numpy()
        for score, target in zip(sub_10_scores, targets):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 / (np.where(score == target - 1)[0][0] + 1))

        sub_5_scores = scores_item.topk(5)[1]
        sub_5_scores = trans_to_cpu(sub_5_scores).detach().numpy()
        for score, target in zip(sub_5_scores, targets):
            hit_5.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_5.append(0)
            else:
                mrr_5.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit_20) * 100)
    result.append(np.mean(mrr_20) * 100)

    result.append(np.mean(hit_10) * 100)
    result.append(np.mean(mrr_10) * 100)

    result.append(np.mean(hit_5) * 100)
    result.append(np.mean(mrr_5) * 100)

    return result
