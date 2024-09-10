
import torch
import torch.nn as nn
from models.attention import MultiHeadAttention

class Attribute_cosmetics(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.w_1 = nn.Parameter(torch.Tensor(args.attribute_dim * 1, args.attribute_dim))
        self.w_2 = nn.Parameter(torch.Tensor(args.attribute_dim * 2, args.attribute_dim))
        self.w_3 = nn.Parameter(torch.Tensor(args.attribute_dim * 2, args.attribute_dim))

        self.w_4 = nn.Parameter(torch.Tensor(args.attribute_dim, 100))

        self.attentions = MultiHeadAttention(model_dim=args.attribute_dim, num_heads=args.num_heads, dropout=0.2)


    def forward(self, seqs, attributes, masks, cat_embedding, brand_embedding):
        cats, brands = attributes
        cat_feat = cat_embedding(cats)
        brand_feat = brand_embedding(brands)
        attr_feat = torch.matmul(torch.cat([cat_feat, brand_feat], -1), self.w_3)

        attr_feat = self.attentions(attr_feat, attr_feat, attr_feat, masks) 
        '''for block in self.transformer_block:
            his_vectors = block(attr_feat, masks)'''

        
        attr_feat = torch.matmul(attr_feat, self.w_4)
        attr_feat = torch.tanh(attr_feat) 

        return attr_feat