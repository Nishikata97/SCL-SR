import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.sparse

from models.item_conv import ItemConv
from models.attribute_cosmetics import Attribute_cosmetics
from models.contrastive import relation_nce_loss, info_nce_loss_overall


class Model_cosmetics(nn.Module):
    def __init__(self, args, adjacency):
        super(Model_cosmetics, self).__init__()
        self.dataset = args.dataset
        self.item_dim = args.item_dim
        self.batch_size = args.batch_size
        self.num_items = args.num_items
        self.L2 = args.l2
        self.lr = args.lr

        self.adjacency = adjacency
        self.ItemGraph = ItemConv(layers=args.num_conv_layers, item_dim=self.item_dim)
        self.Att_encoder = Attribute_cosmetics(args)
        
        self.embedding = nn.Embedding(args.num_items, self.item_dim)
        self.cat_embedding = nn.Embedding(args.num_cats, args.attribute_dim)
        self.brand_embedding = nn.Embedding(args.num_brands, args.attribute_dim)
        self.pos_embedding = nn.Embedding(200, self.item_dim)
        
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.item_dim, self.item_dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.item_dim, 1))
        self.glu1 = nn.Linear(self.item_dim, self.item_dim)
        self.glu2 = nn.Linear(self.item_dim, self.item_dim, bias=False)
        
        self.tau = args.tau
        self.num_coarse_sampling = args.num_coarse_sampling
        self.num_expanded_samples = args.num_expanded_samples
        self.map_dense = nn.Parameter(torch.Tensor(self.item_dim, self.item_dim))
        self.w_3 = nn.Parameter(torch.Tensor(self.item_dim, args.attribute_dim))
        
        self.inter_dense1 = nn.Linear(self.item_dim, self.item_dim)
        self.inter_dense2 = nn.Linear(2 * self.item_dim, self.item_dim)
        self.inter_dense3 = nn.Linear(2 * self.item_dim, self.item_dim)
        self.attr_dense1 = nn.Linear(self.item_dim, self.item_dim)
        self.attr_dense2 = nn.Linear(2 * self.item_dim, self.item_dim)
        self.attr_dense3 = nn.Linear(2 * self.item_dim, self.item_dim)
        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.w_recon = nn.Parameter(torch.Tensor(2 * self.item_dim, self.item_dim))

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.item_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_batch_item_embeddings(self, all_item_embeddings, inputs):
        zeros = torch.cuda.FloatTensor(1, self.item_dim).fill_(0)
        item_embedding = torch.cat([zeros, all_item_embeddings], 0)
        get = lambda i: item_embedding[inputs[i]]
        seq_h = torch.cuda.FloatTensor(inputs.shape[0], list(inputs.shape)[1], self.item_dim).fill_(0)
        for i in torch.arange(inputs.shape[0]):
            seq_h[i] = get(i)
        return seq_h

    def generate_sess_emb(self, seq_h, mask):
        mask = mask.float().unsqueeze(-1)
        batch_size = seq_h.shape[0]
        len = seq_h.shape[1]

        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(seq_h * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.tanh(torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1))
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        sess_embedd = torch.sum(beta * seq_h, 1)
        return sess_embedd

    def generate_sess_emb_npos(self, seq_h, mask):
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]

        hs = torch.sum(seq_h * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        sess_embedd = torch.sum(beta * seq_h, 1)
        return sess_embedd

    def extract_topk_samples(self, original_sampled_items, sample_items_embedd, all_items_embedd):
        len_sess = sample_items_embedd.shape[1]
        num_samples = sample_items_embedd.shape[2]
        sim_items = torch.matmul(sample_items_embedd, all_items_embedd.transpose(1, 0))  
        sim_items = torch.softmax(sim_items, -1)
        values, pos_items = sim_items.topk(self.num_coarse_sampling, dim=-1, largest=True, sorted=True)  

        
        random_slices = torch.randint(low=self.num_expanded_samples, high=self.num_coarse_sampling, size=(self.num_expanded_samples,), device='cuda')
        neg_items_embedd = all_items_embedd[pos_items[:, :, :, random_slices]]  
        neg_items_embedd = neg_items_embedd.view(sample_items_embedd.shape[0], len_sess, -1, self.item_dim)  
        return neg_items_embedd

    def space_mapping(self, inter_feat, attr_feat, substi_items_embedd, comple_items_embedd,
                      expanded_viewed_items_embedd, expanded_bought_items_embedd, inter_sess_embedd, attr_sess_embedd):
        inter_feat = torch.matmul(inter_feat, self.map_dense)
        attr_feat = torch.matmul(attr_feat, self.map_dense)
        substi_items_embedd = torch.matmul(substi_items_embedd, self.map_dense)
        comple_items_embedd = torch.matmul(comple_items_embedd, self.map_dense)
        expanded_viewed_items_embedd = torch.matmul(expanded_viewed_items_embedd, self.map_dense)
        expanded_bought_items_embedd = torch.matmul(expanded_bought_items_embedd, self.map_dense)
        inter_sess_embedd = torch.matmul(inter_sess_embedd, self.map_dense)
        attr_sess_embedd = torch.matmul(attr_sess_embedd, self.map_dense)

        return inter_feat, attr_feat, substi_items_embedd, comple_items_embedd, expanded_viewed_items_embedd, expanded_bought_items_embedd, inter_sess_embedd, attr_sess_embedd

    def loss_predict_attribute(self, inter_feat, all_cat_embedding, all_brand_embedding, cats, brands, mask):
        inter_feat1 = F.normalize(inter_feat, p=2, dim=-1)
        scores_cat = torch.matmul(inter_feat1.view(-1, 300), all_cat_embedding.transpose(1, 0)) 
        scores_brand = torch.matmul(inter_feat1.view(-1, 300), all_brand_embedding.transpose(1, 0))

        label_cat = cats.view(-1)
        label_brand = brands.view(-1)

        mask = mask.float().view(-1) 
        mask = mask == 1
        label_cat = torch.masked_select(label_cat, mask)
        label_brand = torch.masked_select(label_brand, mask)

        scores_cat = torch.masked_select(scores_cat, mask.unsqueeze(-1).repeat(1, all_cat_embedding.shape[0]))
        scores_brand = torch.masked_select(scores_brand, mask.unsqueeze(-1).repeat(1, all_brand_embedding.shape[0]))
        scores_cat = scores_cat.view(-1, all_cat_embedding.shape[0])
        scores_brand = scores_brand.view(-1, all_brand_embedding.shape[0])

        mask_cat = (label_cat != 0)
        mask_brand = (label_brand != 0)
        scores_cat = torch.masked_select(scores_cat, mask_cat.unsqueeze(-1).repeat(1, all_cat_embedding.shape[0]))
        scores_brand = torch.masked_select(scores_brand, mask_brand.unsqueeze(-1).repeat(1, all_brand_embedding.shape[0]))
        scores_cat = scores_cat.view(-1, all_cat_embedding.shape[0])
        scores_brand = scores_brand.view(-1, all_brand_embedding.shape[0])

        label_cat = torch.masked_select(label_cat, mask_cat)
        label_brand = torch.masked_select(label_brand, mask_brand)

        loss_cat = self.loss_function(scores_cat, label_cat - 1)
        loss_brand = self.loss_function(scores_brand, label_brand - 1)

        return loss_cat, loss_brand

    def inter_disen(self, inter_feat):
        inter_disen_feat_1 = self.leakyrelu(self.inter_dense1(inter_feat))
        inter_disen_feat_2 = self.leakyrelu(self.inter_dense2(torch.cat([inter_disen_feat_1, inter_feat], -1)))
        inter_disen_feat_3 = self.leakyrelu(self.inter_dense3(torch.cat([inter_disen_feat_2, inter_feat], -1)))
        return inter_disen_feat_3

    def attr_disen(self, inter_feat):
        attr_disen_feat_1 = self.leakyrelu(self.attr_dense1(inter_feat))
        attr_disen_feat_2 = self.leakyrelu(self.attr_dense2(torch.cat([attr_disen_feat_1, inter_feat], -1)))
        attr_disen_feat_3 = self.leakyrelu(self.attr_dense3(torch.cat([attr_disen_feat_2, inter_feat], -1)))
        return attr_disen_feat_3

    def forward(self, inputs, targets, masks, attributes, viewed_items, bought_items):
        
        all_item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight[1:])  
        inter_feat = self.get_batch_item_embeddings(all_item_embeddings_i, inputs)

        sess_emb_i = self.generate_sess_emb(inter_feat, masks)
        sess_emb_i = 10 * F.normalize(sess_emb_i, dim=-1, p=2) 
        all_item_embeddings_i = F.normalize(all_item_embeddings_i, dim=-1, p=2) 
        scores_item = torch.mm(sess_emb_i, torch.transpose(all_item_embeddings_i, 1, 0)) 
        loss_rec = self.loss_function(scores_item, targets-1)

        
        cats, brands = attributes
        attr_feat = self.Att_encoder(inputs, attributes, masks, self.cat_embedding, self.brand_embedding) 
        sess_emb_a = self.generate_sess_emb(attr_feat, masks)

        
        viewed_items_embedd = self.embedding(viewed_items)  
        bought_items_embedd = self.embedding(bought_items)  

        expanded_substi_items_embedd = self.extract_topk_samples(viewed_items, viewed_items_embedd, self.embedding.weight[1:])
        expanded_comple_items_embedd = self.extract_topk_samples(bought_items, bought_items_embedd, self.embedding.weight[1:])

        substi_items_embedd = viewed_items_embedd
        comple_items_embedd = bought_items_embedd
        
        inter_feat, attr_feat, substi_items_embedd, comple_items_embedd, expanded_substi_items_embedd, expanded_comple_items_embedd, inter_sess_embedd, attr_sess_embedd = self.space_mapping(
            inter_feat, attr_feat, substi_items_embedd, comple_items_embedd, expanded_substi_items_embedd,
            expanded_comple_items_embedd, sess_emb_i, sess_emb_a)

        disen_inter_feat = self.inter_disen(inter_feat)
        disen_attr_feat = self.attr_disen(inter_feat)

        disen_attr_feat_1 = torch.matmul(disen_attr_feat, self.w_3)  
        loss_cat, loss_brand = self.loss_predict_attribute(disen_attr_feat_1, self.cat_embedding.weight[1:],
                                                                        self.brand_embedding.weight[1:], cats,
                                                                        brands, masks)

        loss_attribute = loss_cat + loss_brand

        original_item_embeddings = self.embedding(inputs)
        recon_inter_feat = torch.matmul(torch.cat([disen_inter_feat, disen_attr_feat], dim=-1), self.w_recon)
        loss_reconstruction = F.mse_loss(recon_inter_feat, original_item_embeddings)

        disen_inter_comple_items_embedd = self.inter_disen(comple_items_embedd)
        disen_inter_expanded_substi_items_embedd = self.inter_disen(expanded_substi_items_embedd)
        disen_attr_substi_items_embedd = self.attr_disen(substi_items_embedd)
        disen_attr_expanded_comple_items_embedd = self.attr_disen(expanded_comple_items_embedd)

        loss1 = relation_nce_loss(disen_inter_feat, disen_inter_comple_items_embedd,
                                  disen_inter_expanded_substi_items_embedd, masks, self.tau)

        loss2 = relation_nce_loss(attr_feat, disen_attr_substi_items_embedd, disen_attr_expanded_comple_items_embedd,
                                  masks, self.tau)
        loss_contra = loss1 + loss2
        
        
        disen_attr_sess_embedd = self.generate_sess_emb_npos(disen_attr_feat, masks)
        loss_batch_sess_1 = info_nce_loss_overall(disen_attr_sess_embedd, attr_sess_embedd, disen_attr_sess_embedd, self.tau)
        loss_batch_sess_2 = info_nce_loss_overall(attr_sess_embedd, disen_attr_sess_embedd, attr_sess_embedd, self.tau)
        loss_batch_sess = loss_batch_sess_1 + loss_batch_sess_2

        '''
        loss1 = relation_nce_loss(inter_feat, comple_items_embedd, expanded_substi_items_embedd, masks, self.tau)
        loss2 = relation_nce_loss(attr_feat, substi_items_embedd, expanded_comple_items_embedd, masks, self.tau)
        loss_contra = loss1 + loss2'''

        loss_batch_sess = 0

        return scores_item, loss_rec, loss_attribute, loss_reconstruction, loss_contra, loss_batch_sess
