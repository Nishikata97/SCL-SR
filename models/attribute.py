
import torch
import torch.nn as nn
from models.attention import MultiHeadAttention
from models.transformer import TransformerEncoder

class Attribute(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.w_1 = nn.Parameter(torch.Tensor(args.attribute_dim * 1, args.attribute_dim))
        self.w_2 = nn.Parameter(torch.Tensor(args.attribute_dim * 2, args.attribute_dim))
        self.w_3 = nn.Parameter(torch.Tensor(args.attribute_dim * 3, args.attribute_dim))

        self.w_4 = nn.Parameter(torch.Tensor(args.attribute_dim, 100))

        self.attentions = MultiHeadAttention(model_dim=args.attribute_dim, num_heads=3, dropout=0.5)

        '''self.transformer = TransformerEncoder(n_layers=5, n_heads=4, hidden_size=300, inner_size=256,
                                              hidden_dropout_prob=0.5, attn_dropout_prob=0.5, hidden_act="gelu",
                                              layer_norm_eps=1e-12, )'''

    def forward(self, seqs, attributes, masks, cat_embedding, seller_embedding,brand_embedding):
        if self.args.dataset in ['Tmall', 'CIKM19']:
            cats, sellers, brands = attributes
            cat_feat = cat_embedding(cats)
            seller_feat = seller_embedding(sellers)
            brand_feat = brand_embedding(brands)
            attr_feat = torch.matmul(torch.cat([cat_feat, seller_feat, brand_feat], -1), self.w_3)
        elif self.args.dataset == 'Cosmetics':
            cats, brands = attributes
            cat_feat = cat_embedding(cats)
            brand_feat = brand_embedding(brands)
            attr_feat = torch.matmul(torch.cat([cat_feat, brand_feat], -1), self.w_2)
        else:
            cats = attributes
            cat_feat = cat_embedding(cats)
            attr_feat = torch.matmul(cat_feat, self.w_1)

        '''masks = masks.unsqueeze(1).unsqueeze(2)
        attr_feat = self.transformer(attr_feat, masks)
        attr_feat = attr_feat[0]'''
        attr_feat = self.attentions(attr_feat, attr_feat, attr_feat, masks) 

        
        attr_feat = torch.matmul(attr_feat, self.w_4)
        attr_feat = torch.tanh(attr_feat) 

        return attr_feat