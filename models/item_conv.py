import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

class ItemConv(nn.Module):
    def __init__(self, layers, item_dim=100):
        super(ItemConv, self).__init__()
        self.item_dim = item_dim
        self.layers = layers
        self.w_item = {}
        for i in range(self.layers):
            self.w_item['weight_item%d' % (i)] = nn.Linear(self.item_dim, self.item_dim, bias=False)

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape)) 
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            item_embeddings = trans_to_cuda(self.w_item['weight_item%d' % (i)])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            final.append(F.normalize(item_embeddings, dim=-1, p=2))
        item_embeddings = torch.sum(torch.stack(final), dim=0) / (self.layers + 1)
        return item_embeddings
