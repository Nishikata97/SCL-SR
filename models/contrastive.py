import datetime
from tqdm import tqdm
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sim(z1: torch.Tensor, z2: torch.Tensor):
    if z1.size()[0] == z2.size()[0]:
        return F.cosine_similarity(z1, z2, dim=-1)
    else:
        z1 = F.normalize(z1, p=2, dim=-1)  
        z2 = F.normalize(z2, p=2, dim=-1)
        return torch.mm(z1, z2.t())  



def relation_nce_loss(feat, posi_embedd, nega_embedd, mask, tau):
    batch_size = feat.shape[0]
    dim = feat.shape[-1]
    num_posi = posi_embedd.shape[2]
    num_nega = nega_embedd.shape[2]

    mask = mask.float().view(-1).unsqueeze(-1)  
    mask = mask == 1
    feat = torch.masked_select(feat.view(-1, dim), mask) 
    posi_embedd = torch.masked_select(posi_embedd.view(-1, num_posi, dim), mask.unsqueeze(-1))
    nega_embedd = torch.masked_select(nega_embedd.view(-1, num_nega, dim), mask.unsqueeze(-1))

    feat = feat.view(-1, dim) 
    posi_embedd = posi_embedd.view(-1, num_posi, dim) 
    nega_embedd = nega_embedd.view(-1, num_nega, dim) 

    f = lambda x: torch.exp(x / tau)  

    posi_sim = f(sim(feat.unsqueeze(1).repeat(1, posi_embedd.shape[1], 1), posi_embedd)) 
    nega_sim = f(sim(feat.unsqueeze(1).repeat(1, nega_embedd.shape[1], 1), nega_embedd)) 

    
    

    positive_pairs = torch.sum(posi_sim, -1) 
    negative_pairs = torch.sum(nega_sim, -1) 

    loss = torch.sum(-torch.log(positive_pairs / (positive_pairs + negative_pairs)))
    return loss


def info_nce_loss_overall(z1, z2, z_all, tau):
    f = lambda x: torch.exp(x / tau)  

    between_sim = f(sim(z1, z2))  
    z_all_1 = z_all.transpose(1, 0)  
    z_all = F.normalize(z_all, p=2, dim=-1)
    z_all_1 = F.normalize(z_all_1, p=2, dim=-1)
    all_sim = f(torch.matmul(z_all, z_all_1)) 

    positive_pairs = between_sim
    negative_pairs = torch.sum(all_sim, 1)
    loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
    return loss
