
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = attention_dropout

    def forward(self, q, k, v, scale=None, attn_mask=None):
        
        seq_len = q.shape[-2]
        attribute_dim = q.shape[-1]

        attention = torch.matmul(q.view(-1, seq_len, attribute_dim), k.view(-1, seq_len, attribute_dim).transpose(1, 2)) 
        
        if scale:
            attention = attention * scale 
        

        
        attention = torch.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        output = torch.matmul(attention, v.view(-1, seq_len, attribute_dim)) 
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=300, num_heads=3, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads) 
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads) 
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads) 

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = dropout

        self.pos_embedding = nn.Embedding(100, model_dim)
        self.w_1 = nn.Parameter(torch.Tensor(2 * model_dim, model_dim))
        self.w_2 = nn.Parameter(torch.Tensor(model_dim, 1))
        self.glu1 = nn.Linear(model_dim, model_dim)
        self.glu2 = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, query, key, value, mask=None):
        residual = query 

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.shape[0]
        items_num = key.shape[1]
        review_len = key.shape[2]

        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        scale = (key.shape[-1] // num_heads) ** -0.5
        context = self.dot_product_attention(query, key, value, scale, mask)
        
        output = self.linear_final(context)
        output = F.dropout(output, self.dropout, training=self.training) 
        output = residual + output 

        return output 

