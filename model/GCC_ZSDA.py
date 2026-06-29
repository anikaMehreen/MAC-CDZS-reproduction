'''
Graph SAGE append
'''
import math
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

import pickle
import random
import model.networks as networks
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import scipy.io as sio
from functools import partial
from sklearn import metrics
from einops.layers.torch import Rearrange, Reduce
# # import torch_clustering
from torch.utils.tensorboard import SummaryWriter

from sklearn.mixture import GaussianMixture
torch.cuda.set_device(0)
import random
class UnsupervisedLoss(object):
    """docstring for UnsupervisedLoss"""

    def __init__(self, adj_lists, train_nodes, device):
        super(UnsupervisedLoss, self).__init__()
        self.Q = 10
        self.N_WALKS = 6
        self.WALK_LEN = 1
        self.N_WALK_LEN = 5
        self.MARGIN = 3
        self.adj_lists = adj_lists
        self.train_nodes = train_nodes
        self.device = device

        self.target_nodes = None
        self.positive_pairs = []
        self.negtive_pairs = []
        self.node_positive_pairs = {}
        self.node_negtive_pairs = {}
        self.unique_nodes_batch = []

    def get_loss_sage(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        # assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            # Q * Exception(negative score)
            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score = self.Q * torch.mean(torch.log(torch.sigmoid(-neg_score)), 0)
            # print(neg_score)

            # multiple positive score
            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score = torch.log(torch.sigmoid(pos_score))
            # print(pos_score)

            nodes_score.append(torch.mean(- pos_score - neg_score).view(1, -1))

        loss = torch.mean(torch.cat(nodes_score, 0))

        return loss

    def get_loss_margin(self, embeddings, nodes):
        assert len(embeddings) == len(self.unique_nodes_batch)
        assert False not in [nodes[i] == self.unique_nodes_batch[i] for i in range(len(nodes))]
        node2index = {n: i for i, n in enumerate(self.unique_nodes_batch)}

        nodes_score = []
        assert len(self.node_positive_pairs) == len(self.node_negtive_pairs)
        for node in self.node_positive_pairs:
            pps = self.node_positive_pairs[node]
            nps = self.node_negtive_pairs[node]
            if len(pps) == 0 or len(nps) == 0:
                continue

            indexs = [list(x) for x in zip(*pps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            pos_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            pos_score, _ = torch.min(torch.log(torch.sigmoid(pos_score)), 0)

            indexs = [list(x) for x in zip(*nps)]
            node_indexs = [node2index[x] for x in indexs[0]]
            neighb_indexs = [node2index[x] for x in indexs[1]]
            neg_score = F.cosine_similarity(embeddings[node_indexs], embeddings[neighb_indexs])
            neg_score, _ = torch.max(torch.log(torch.sigmoid(neg_score)), 0)

            nodes_score.append(
                torch.max(torch.tensor(0.0).to(self.device), neg_score - pos_score + self.MARGIN).view(1, -1))
        # nodes_score.append((-pos_score - neg_score).view(1,-1))

        loss = torch.mean(torch.cat(nodes_score, 0), 0)

        # loss = -torch.log(torch.sigmoid(pos_score))-4*torch.log(torch.sigmoid(-neg_score))

        return loss

    def extend_nodes(self, nodes, num_neg=6):
        self.positive_pairs = []
        self.node_positive_pairs = {}
        self.negtive_pairs = []
        self.node_negtive_pairs = {}

        self.target_nodes = nodes
        self.get_positive_nodes(nodes)
        # print(self.positive_pairs)
        self.get_negtive_nodes(nodes, num_neg)
        # print(self.negtive_pairs)
        self.unique_nodes_batch = list(
            set([i for x in self.positive_pairs for i in x]) | set([i for x in self.negtive_pairs for i in x]))
        assert set(self.target_nodes) < set(self.unique_nodes_batch)
        return self.unique_nodes_batch

    def get_positive_nodes(self, nodes):
        return self._run_random_walks(nodes)

    def get_negtive_nodes(self, nodes, num_neg):
        for node in nodes:
            neighbors = set([node])
            frontier = set([node])
            for i in range(self.N_WALK_LEN):
                current = set()
                for outer in frontier:
                    current |= self.adj_lists[int(outer)]
                frontier = current - neighbors
                neighbors |= current
            far_nodes = set(self.train_nodes) - neighbors
            random.seed(123)
            neg_samples = random.sample(list(far_nodes), num_neg) if num_neg < len(far_nodes) else list(far_nodes)
            self.negtive_pairs.extend([(node, neg_node) for neg_node in neg_samples])
            self.node_negtive_pairs[node] = [(node, neg_node) for neg_node in neg_samples]
        return self.negtive_pairs

    def _run_random_walks(self, nodes):
        for node in nodes:
            if len(self.adj_lists[int(node)]) == 0:
                continue
            cur_pairs = []
            for i in range(self.N_WALKS):
                curr_node = node
                for j in range(self.WALK_LEN):
                    neighs = self.adj_lists[int(curr_node)]
                    random.seed(234)
                    next_node = random.choice(list(neighs))
                    # self co-occurrences are useless
                    if next_node != node and next_node in self.train_nodes:
                        self.positive_pairs.append((node, next_node))
                        cur_pairs.append((node, next_node))
                    curr_node = next_node

            self.node_positive_pairs[node] = cur_pairs
        return self.positive_pairs

class SageLayer(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """

    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(self, self_feats, aggregate_feats, neighs=None):
        """
        Generates embeddings for a batch of nodes.

        nodes	 -- list of nodes
        """
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats
        combined = F.relu(self.weight.mm(combined.t())).t()
        return combined

class GraphSage(nn.Module):
    """docstring for GraphSage"""

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, device, raw_agu1_features=None, raw_aug2_features=None, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.device = device
        self.agg_func = agg_func

        self.raw_features = raw_features
        self.raw_agu1_features = raw_agu1_features
        self.raw_aug2_features = raw_aug2_features
        self.adj_lists = adj_lists

        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def forward_aug1(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_agu1_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def forward_aug2(self, nodes_batch):
        """
        Generates embeddings for a batch of nodes.
        nodes_batch	-- batch of nodes to learn the embeddings
        """
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        # self.dc.logger.info('get_unique_neighs.')
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes)
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_aug2_features
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]
            # self.dc.logger.info('aggregate_feats.')
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            sage_layer = getattr(self, 'sage_layer' + str(index))
            if index > 1:
                nb = self._nodes_map(nb, pre_hidden_embs, pre_neighs)
            # self.dc.logger.info('sage_layer.')
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, hidden_embs, neighs):
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, num_sample=10):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        if not num_sample is None:
            random.seed(345)
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample)) if len(to_neigh) >= num_sample else to_neigh for to_neigh
                           in to_neighs]
        else:
            samp_neighs = to_neighs
        samp_neighs = [samp_neigh | set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))
        unique_nodes = dict(list(zip(_unique_nodes_list, i)))
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs, num_sample=10):
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - set([nodes[i]])) for i in range(len(samp_neighs))]
        # self.dc.logger.info('2')
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]
        # self.dc.logger.info('3')
        # General fix: sparse index-based aggregation instead of a dense (nodes x unique_nodes)
        # mask matrix. Memory now scales with the number of actual edges, not nodes^2 --
        # so it's safe regardless of dataset, seed, or how many nodes a batch expands to.
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        row_idx_t = torch.tensor(row_indices, dtype=torch.long, device=embed_matrix.device)
        col_idx_t = torch.tensor(column_indices, dtype=torch.long, device=embed_matrix.device)

        if self.agg_func == 'MEAN':
            neighbor_feats = embed_matrix[col_idx_t]  # (num_edges, feat_dim)
            out_dim = embed_matrix.shape[1]

            agg_sum = torch.zeros(len(samp_neighs), out_dim, device=embed_matrix.device, dtype=embed_matrix.dtype)
            agg_sum.index_add_(0, row_idx_t, neighbor_feats)

            counts = torch.zeros(len(samp_neighs), device=embed_matrix.device, dtype=embed_matrix.dtype)
            counts.index_add_(0, row_idx_t, torch.ones(len(row_idx_t), device=embed_matrix.device, dtype=embed_matrix.dtype))
            counts = counts.clamp(min=1).unsqueeze(1)  # avoid divide-by-zero for neighborless nodes

            aggregate_feats = agg_sum / counts
        elif self.agg_func == 'MAX':
            row_idx_t = torch.tensor(row_indices, dtype=torch.long, device=embed_matrix.device)
            col_idx_t = torch.tensor(column_indices, dtype=torch.long, device=embed_matrix.device)
            indexs = [torch.where(row_idx_t == i)[0] for i in range(len(samp_neighs))]
            aggregate_feats = []
            for idx in indexs:
                feat = embed_matrix[col_idx_t[idx]]
                if feat.shape[0] == 0:
                    aggregate_feats.append(torch.zeros(1, embed_matrix.shape[1], device=embed_matrix.device))
                elif feat.shape[0] == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        # self.dc.logger.info('6')

        return aggregate_feats

###################################################################################
class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension, init_weights=True):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# MLP
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) #+ x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )
class MLP_Block(nn.Module):
    def __init__(self, dim, depth):
        super(MLP_Block, self).__init__()
        self.dim = dim
        self.depth = depth
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.num_patches = 81
        self.expansion_factor = 4
        self.expansion_factor_token = 2

        # self.lin1 = nn.Linear((1 ** 2) * 100, dim)
        self.block = nn.Sequential(
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),

            nn.Linear((1 ** 2) * self.dim, self.dim),
            *[nn.Sequential(
                PreNormResidual(self.dim, FeedForward(self.num_patches, self.expansion_factor, 0., self.chan_first)),
                PreNormResidual(self.dim, FeedForward(128, self.expansion_factor_token, 0., self.chan_last))
            ) for _ in range(self.depth)],
            nn.LayerNorm(self.dim),
            Reduce('b n c -> b c', 'mean'),
            nn.Linear(self.dim, self.dim),
            Rearrange('b (c p1 p2) -> b c p1 p2', p1=1, p2=1)
        )
    def forward(self, x):
        return self.block(x)

class MLP_Mapping_Block(nn.Module):
    def __init__(self, dim, depth, channel):
        super(MLP_Mapping_Block, self).__init__()
        self.dim = dim
        self.depth = depth
        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.channel = channel
        self.num_patches = 81
        self.expansion_factor = 2
        self.expansion_factor_token = 2

        self.block = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            nn.Linear((1 ** 2) * self.channel, self.dim),
            *[nn.Sequential(
                PreNormResidual(self.dim, FeedForward(self.num_patches, self.expansion_factor, 0., self.chan_first)),
                PreNormResidual(self.dim, FeedForward(self.dim, self.expansion_factor_token, 0., self.chan_last))
            ) for _ in range(self.depth)],
            nn.LayerNorm(self.dim),
            # Reduce('b n c -> b c', 'mean'),
            # nn.Linear(self.dim, self.dim),
            # Rearrange('b (c p1 p2) -> b c p1 p2', p1=1, p2=1)
        )
    def forward(self, x):
        return self.block(x)
###################



class ZSDAModel():
    def __init__(self, tar, IR_task, R_task, IR_list, R_list, T_r_data, T_r_labels, adj_lists, path=None, channel=None):
        self.gpu_ids = [0]
        self.patchsize = 9
        self.rt_class = R_task
        self.irt_class = IR_task
        self.rt_list = R_list
        self.irt_list = IR_list
        self.lr = 0.0002  # Initial learning rate for adam optimizer
        self.beta1 = 0.5  # Momentum term of adam
        self.lr_policy = 'step'
        self.tardet = tar
        self.n_epochs = 10000
        self.epoch = 0
        self.model_names = ['source_mapping', 'target_mapping', 'G_D', 'G_T', 'D_D', 'FS', 'C_R', 'C_IR', 'projector']
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))

        self.chan_first, self.chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear
        self.rt_s_proto = torch.zeros([self.rt_class, 128])
        self.rt_t_proto = torch.zeros([self.rt_class, 128])
        self.rt_s2t_list = torch.zeros([self.rt_class, ])
        self.sim_matrix = torch.zeros([self.rt_class, self.rt_class])
        self.T_r_data = torch.tensor(T_r_data).to(self.device)
        self.T_r_labels = torch.tensor(T_r_labels).to(self.device)

        # finetune
        self.T = 0.007
        self.num_cluster = R_task
        self.best_rt_acc = 0.0
        self.warmup = False
        self.in_dim = 128
        self.fea_dim = 128
        self.mixup_alpha = 1.0
        self.scale1 = 0.
        self.scale2 = 1.
        self.temp = 0.25
        self.num_samples = 128
        self.prototypes = torch.randn(self.num_cluster, 64)  # fixed: must match GraphSAGE's 64-dim output, not R_task
        self.prototypes = F.normalize(self.prototypes, dim=1)
        self.confidences = []
        self.context_assignments = torch.zeros([128*6, R_task])
        self.fintuning_test_features = []
        self.fintuning_features = []
        self.fintuning_clean_labels = []

        # `channel` (spectral band count) now comes from dataset_config.py
        # via the `channel` argument, instead of being hardcoded per
        # dataset name here. The old if/elif chain below only runs as a
        # fallback if this model is ever constructed without passing
        # `channel` explicitly (e.g. by old code that hasn't been updated).
        #
        # `Embedding` has been removed entirely: every reference to the
        # nn.Embedding layer that used it lives inside init_Graph(), which
        # is fully commented out elsewhere in this file -- confirmed via
        # grep that no other method reads self.features. It was dead code.
        if channel is None:
            channel = 100
            if self.tardet == 'UP':
                channel = 103
            elif self.tardet == 'PC':
                channel = 102
            elif self.tardet == 'SA':
                channel = 204
            elif self.tardet == 'IP':
                channel = 200
            elif self.tardet == 'Ho':
                channel = 144
            elif self.tardet == 'Ut':
                channel = 432
            elif self.tardet == 'MS':
                channel = 44
        Embedding = 1  # dead code, kept only because nn.Embedding needs *some* size

        # Graph SAGE set
        self.GCN_layers = 5
        self.feat_data = None
        self.features = nn.Embedding(Embedding, 128)
        self.adj_lists = adj_lists
        # self.features.weight = nn.Parameter(torch.FloatTensor(self.feat_data), requires_grad=False)
        #################

        log_path = './log/all_loss'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.writer = SummaryWriter(log_path)

        # mapping layer

        # self.nettarget_mapping = networks.init_net(Mapping(channel, 100), gpu_ids=self.gpu_ids)  # PU 103  PC:102 sa: 204 IP:200
        # self.netsource_mapping = networks.init_net(Mapping(128, 100), gpu_ids=self.gpu_ids)  # chikusei 128

        self.nettarget_mapping = networks.init_net(MLP_Mapping_Block(dim=128, depth=1, channel=channel)).to(self.device)
        self.netsource_mapping = networks.init_net(MLP_Mapping_Block(dim=128, depth=1, channel=128)).to(self.device)

        # feature encoder  G_d  and  G_c
        avgPool = False
        # self.netG_D = networks.init_net(networks.FeatureExtractor(input_nc=100, output_nc=256, avgPool=avgPool),
        #                                 gpu_ids=self.gpu_ids)
        # self.netG_T = networks.init_net(networks.FeatureExtractor(input_nc=100, output_nc=256, avgPool=avgPool),
        #                                 gpu_ids=self.gpu_ids)
        self.netG_D = networks.init_net(MLP_Block(128, 2)).to(self.device)
        self.netG_T = networks.init_net(MLP_Block(128, 2)).to(self.device)

        # feature_size = 128 * (round(round(self.patchsize / 2) / 2)) ** 2
        # nc = 128
        # feature_size = 256*16
        feature_size = 128
        # nc = 256
        nc = 128
        resnet = False

        self.GRL = networks.init_net(networks.GRL())
        self.netD_D = networks.init_net(networks.Discriminator(input_nc=feature_size, resnet=resnet),
                                        gpu_ids=self.gpu_ids)
        self.netC_R = networks.init_net(
            networks.Classifier(input_nc=feature_size, output_nc=self.rt_class, resnet=resnet), gpu_ids=self.gpu_ids)
        self.netC_IR = networks.init_net(
            networks.Classifier(input_nc=feature_size, output_nc=self.irt_class, resnet=resnet), gpu_ids=self.gpu_ids)
        self.pool = networks.init_net(nn.AdaptiveAvgPool2d(1), gpu_ids=self.gpu_ids)

        self.fs_loss_coeff = {'irt_s': 1, 'irt_t': 1, 'rt_s': 2}
        self.cls_loss_coeff = {'irt_s': 0, 'irt_t': 0, 'rt_s': 2}

        # self.netFS = networks.init_net(networks.FeatureShifter_Att_FC(input_nc=nc), gpu_ids=self.gpu_ids)
        self.netFS = networks.init_net(networks.FeatureShifter_Att(input_nc=nc), gpu_ids=self.gpu_ids)

        self.netprojector = networks.init_net(
            networks.Projector(input_cn=feature_size, output_cn=self.rt_class), gpu_ids=self.gpu_ids)
            # networks.Classifier(input_nc=feature_size, output_nc=self.rt_class, resnet=resnet), gpu_ids=self.gpu_ids)

        self.optG_T = torch.optim.Adam(self.netG_T.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optG_D = torch.optim.Adam(self.netG_D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optD_D = torch.optim.Adam(self.netD_D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optC_R = torch.optim.Adam(self.netC_R.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optC_IR = torch.optim.Adam(self.netC_IR.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optFS = torch.optim.Adam(self.netFS.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optsource_mapping = torch.optim.Adam(self.netG_T.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.opttarget_mapping = torch.optim.Adam(self.netG_D.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optprojector = torch.optim.Adam(self.netprojector.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.optimizers = self._get_opts(self.model_names)
        self.schedulers = [networks.get_scheduler(optimizer, 'step') for optimizer in self.optimizers]


        self.criterion_xent = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.criterion_adv = nn.BCEWithLogitsLoss().to(self.device)

        self.losses = {}
        self.counts = {}

        from .utils.infonce import InstanceLoss
        self.loss = InstanceLoss(0.25)

        # self.save_dir = os.path.join(cfg.checkpoints_dir, cfg.name)
        self.save_dir = path #'./ckpt_GCC/IP'
        # if not os.path.exists(self.save_dir):
        #     os.makedirs(self.save_dir)


    # def init_Graph(self):
    #     self.features.weight = nn.Parameter(torch.FloatTensor(self.feat_data), requires_grad=False)
    #     agg1 = MeanAggregator(self.features, cuda=True)
    #     enc1 = Encoder(self.features, 128, 128, self.adj_lists, agg1, gcn=True, cuda=False)
    #     agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    #     enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, self.adj_lists, agg2,
    #                    base_model=enc1, gcn=True, cuda=False)
    #     enc1.num_samples = 5
    #     enc2.num_samples = 5
    #
    #     self.netgraphsage = SupervisedGraphSage(2, enc2)
    #     self.optC_R = torch.optim.Adam(self.netgraphsage.parameters(), lr=self.lr, betas=(self.beta1, 0.999))




    def set_hyper_input(self, irt_s, irt_s_lab, irt_t, irt_t_lab, rt_s, rt_s_lab):
        self.x = {'irt_s': irt_s.to(self.device),
                  'irt_t': irt_t.to(self.device),
                  'rt_s': rt_s.to(self.device)}
        self.y = {'irt_s': irt_s_lab.to(self.device),
                  'irt_t': irt_t_lab.to(self.device),
                  'rt_s': rt_s_lab.to(self.device)}


    def set_input(self, irt_s, irt_t, rt_s):
            self.x = {'irt_s': irt_s[0].to(self.device),
                      'irt_t': irt_t[0].to(self.device),
                      'rt_s': rt_s[0].to(self.device)}
            self.y = {'irt_s': irt_s[1].to(self.device),
                      'irt_t': irt_t[1].to(self.device),
                      'rt_s': rt_s[1].to(self.device)}


    def set_pair_input(self, irt_s, irt_t, rt_s):
        self.B = irt_s[0]['anchor'].shape[0]
        irt_s_x = torch.cat((irt_s[0]['anchor'], irt_t[0]['negative']), 0)
        irt_s_y = torch.cat((irt_s[1]['anchor'], irt_t[1]['negative']), 0)
        irt_t_x = torch.cat((irt_s[0]['positive'], irt_t[0]['anchor']), 0)
        irt_t_y = torch.cat((irt_s[1]['positive'], irt_t[1]['anchor']), 0)

        self.x = {'irt_s': irt_s_x.to(self.device),
                  'irt_t': irt_t_x.to(self.device),
                  'rt_s': rt_s[0].to(self.device)}
        self.y = {'irt_s': irt_s_y.to(self.device),
                  'irt_t': irt_t_y.to(self.device),
                  'rt_s': rt_s[1].to(self.device)}

    def _get_opts(self, model_names):
        opts = []
        for name in model_names:
            opt = getattr(self, 'opt' + name)
            opts.append(opt)

        return opts

    def update(self):
        # Old version
        # self.class_disentangle()
        # self.update_FS()
        # Paper version


        loss1 = self.domain_disentangle()
        # self.domain_disentangle()
        loss2, loss3 = self.class_disentangle()
        # loss2, loss3 = loss1, loss1
        loss4 = self.collab_learning()

        return loss1, loss2, loss3, loss4

    def domain_disentangle(self):
        updated_models = ['C_R', 'C_IR', 'D_D', 'G_T', 'G_D', 'source_mapping', 'target_mapping']
        # updated_models = ['C_R', 'C_IR', 'D_D', 'G_T', 'G_D']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        self.x['irt_s'] = self.netsource_mapping(self.x['irt_s'])
        self.x['irt_t'] = self.nettarget_mapping(self.x['irt_t'])
        self.x['rt_s'] = self.netsource_mapping(self.x['rt_s'])

        loss1 = 0
        for task in self.x:
            f_ci = self.netG_D(self.x[task])
            f_di = self.netG_T(self.x[task])

            task_loss = 0
            if task == 'rt_s':
                di_class_pred = self.netC_R(f_di)
            else:
                di_class_pred = self.netC_IR(f_di)

            #  classification loss
            task_loss += self.criterion_xent(di_class_pred, self.y[task]) * self.fs_loss_coeff[task] / 3

            domain_pred = self.netD_D(self.GRL(f_di))
            ci_domain_pred = self.netD_D(f_ci)

            if task == 'irt_t':
                domain_loss = self.criterion_adv(ci_domain_pred, torch.ones(ci_domain_pred.shape).to(self.device)) + \
                              self.criterion_adv(domain_pred, torch.ones(domain_pred.shape).to(self.device))
                # print('irt_t domain loss:', domain_loss)
            else:
                domain_loss = self.criterion_adv(ci_domain_pred, torch.zeros(ci_domain_pred.shape).to(self.device)) + \
                              self.criterion_adv(domain_pred, torch.zeros(domain_pred.shape).to(self.device))
                # print(task+' domain loss:', domain_loss)

            loss1 += task_loss + domain_loss / 3

        loss1.backward(retain_graph=True)

        for opt in opts:
            opt.step()

        return loss1

    def class_disentangle(self):
        updated_models = ['C_R', 'C_IR']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss2 = 0
        for task in self.x:
            feat = self.netG_D(self.x[task]).detach()
            if task == 'rt_s':
                class_pred = self.netC_R(feat)
            else:
                class_pred = self.netC_IR(feat)

            loss2 += self.criterion_xent(class_pred, self.y[task]) / 3

        loss2.backward(retain_graph=True)

        for opt in opts:
            opt.step()

        #  G_D
        updated_models = ['G_D']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss3 = 0
        for task in self.x:
            feat = self.netG_D(self.x[task])
            if task == 'rt_s':
                class_pred = self.netC_R(feat)
            else:
                class_pred = self.netC_IR(feat)

            loss3 += - torch.mean(torch.log(torch.nn.functional.softmax(class_pred + 1e-6, dim=-1))) / 3

        loss3.backward(retain_graph=True)

        for opt in opts:
            opt.step()

        return loss2, loss3

    def collab_learning(self):
        updated_models = ['G_T', 'G_D', 'FS', 'C_R', 'C_IR', 'source_mapping', 'target_mapping']
        # updated_models = ['G_T', 'G_D', 'FS', 'C_R', 'C_IR']
        opts = self._get_opts(updated_models)
        for opt in opts:
            opt.zero_grad()

        loss4 = 0

        for task in self.x:
            f_ci = self.netG_D(self.x[task])
            f_di = self.netG_T(self.x[task])
            feat = self.netFS(f_ci, f_di)

            if task == 'rt_s':
                class_pred = self.netC_R(feat)
            else:
                class_pred = self.netC_IR(feat)
            loss4 += self.criterion_xent(class_pred, self.y[task]) * self.fs_loss_coeff[task]

        loss4.backward(retain_graph=True)

        for opt in opts:
            opt.step()
        return loss4


    def test(self, r_test_loader, r_task, ir_test_loader, ir_task, rt_s_data, rt_s_lab, SELF_s2t=None):
        models = [self.netG_D, self.netG_T, self.netFS, self.netC_R, self.netC_IR, self.netprojector]
        for model in models:
            model.eval()
        di_correct = 0
        correct = 0
        count, count_ir = 0, 0

        IR_predict = np.array([], dtype=np.int64)
        IR_labels = np.array([], dtype=np.int64)
        R_predict = np.array([], dtype=np.int64)
        R_labels = np.array([], dtype=np.int64)

        re_fea = torch.tensor(np.array([], dtype=np.int64))
        re_data = torch.tensor(np.array([], dtype=np.int64))

        # ALL_predict = np.array([], dtype=np.int64)
        # ALL_labels = np.array([], dtype=np.int64)

        if SELF_s2t is None:
            ##### save rt_s mean features
            print('rt_s')
            for i in range(self.rt_class):
                index = torch.where(rt_s_lab == i)[0]
                da = rt_s_data[index, :]

                cla_name = self.rt_list[i]
                index_t = torch.where(self.T_r_labels == torch.tensor(cla_name).to(self.device))[0]
                da_t = self.T_r_data[index_t, ]
                with torch.no_grad():
                    da = da.to(self.device)
                    da = self.netsource_mapping(da)
                    f_ci, f_di = self.netG_D(da), self.netG_T(da)
                    feat = self.netFS(f_ci, f_di)

                    da_t = da_t.to(self.device)
                    da_t = self.nettarget_mapping(da_t)
                    f_ci_t, f_di_t = self.netG_D(da_t), self.netG_T(da_t)
                    feat_t = self.netFS(f_ci_t, f_di_t)
                mean = torch.squeeze(torch.mean(feat, dim=0))
                self.rt_s_proto[i, ] = mean

                mean_t = torch.squeeze(torch.mean(feat_t, dim=0))
                self.rt_t_proto[i, ] = mean_t

            # simalarity measure
            for i in range(self.rt_class):
                sim = torch.cosine_similarity(torch.unsqueeze(self.rt_s_proto[i, ], dim=0), self.rt_t_proto, dim=1)
                # print(sim)
                self.sim_matrix[:, i] = sim
                max_index = torch.argmax(sim)
                # self.rt_s2t_list[i] = max_index
            print('rt_s2t_list: ', self.rt_s2t_list)
            # print('sim matrix: ')
            # print(self.sim_matrix)
            s2t = torch.zeros([self.rt_class, ])
            s2t = s2t - 1
            ma = self.sim_matrix.reshape([self.rt_class ** 2, 1])
            max_index = torch.argsort(ma, dim=0, descending=True)
            cnt = 0
            i = 0
            while cnt < self.rt_class:
                id = max_index[i]
                row = id // self.rt_class
                clo = id % self.rt_class
                if s2t[clo] == -1:
                    if row not in s2t:
                        s2t[clo] = int(row)
                        cnt += 1
                        i += 1
                    else:
                        i += 1
                else:
                    i += 1
            self.rt_s2t_list = s2t
        else:
            self.rt_s2t_list = SELF_s2t
        # print('rt_s2t_list: ', self.rt_s2t_list)

        #####
        # R-task
        flag = 0
        with torch.no_grad():
            for x, y, row, clo in r_test_loader:
                x = x.to(self.device)
                # if flag == 0:
                #     re_data = x[:, :, 4, 4].cpu()
                # else:
                #     re_data = torch.cat([re_data, x[:, :, 4, 4].cpu()], dim=0)
                x = self.nettarget_mapping(x)

                f_ci, f_di = self.netG_D(x), self.netG_T(x)
                if r_task == 'rt' or r_task == 'rs':
                    di_pred = self.netC_R(f_di)
                elif r_task == 'irt' or r_task == 'irs':
                    di_pred = self.netC_IR(f_di)

                # di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y).item()
                di_lab = torch.argmax(di_pred, dim=1)
                # for la_i in range(len(self.rt_list)):
                #     di_lab[di_lab == la_i] = self.rt_list[la_i]

                feat = self.netFS(f_ci, f_di)
                if r_task == 'rt' or r_task == 'rs':
                    pred = self.netC_R(feat)
                elif r_task == 'irt' or r_task == 'irs':
                    pred = self.netC_IR(feat)

                # feature store for Graph
                if flag == 0:
                    self.feat_data = feat.squeeze()
                    flag = 1
                else:
                    self.feat_data = torch.cat([self.feat_data, feat.squeeze()], dim=0)
                #

                p_lab = torch.argmax(pred, dim=1)
                # for la_i in range(len(self.rt_list)):
                #     p_lab[p_lab == la_i] = self.rt_list[la_i]
                #     di_lab[di_lab == la_i] = self.rt_list[la_i]

                # source to target
                index_list, di_index_list = [], []
                for la_i in range(self.rt_class):
                    ind_i = torch.where(p_lab == la_i)[0]
                    index_list.append(ind_i)
                    di_ind_i = torch.where(di_lab == la_i)[0]
                    di_index_list.append(di_ind_i)
                for la_i in range(len(self.rt_list)):
                    ind_i = index_list[la_i]
                    di_ind_i = di_index_list[la_i]
                    p_lab[ind_i] = int(self.rt_s2t_list[la_i])
                    di_lab[di_ind_i] = int(self.rt_s2t_list[la_i])


                index_list, di_index_list = [], []
                for la_i in range(self.rt_class):
                    ind_i = torch.where(p_lab == la_i)[0]
                    index_list.append(ind_i)
                    di_ind_i = torch.where(di_lab == la_i)[0]
                    di_index_list.append(di_ind_i)
                for la_i in range(len(self.rt_list)):
                    ind_i = index_list[la_i]
                    di_ind_i = di_index_list[la_i]
                    p_lab[ind_i] = self.rt_list[la_i]
                    di_lab[di_ind_i] = self.rt_list[la_i]

                ##########################################################
                R_predict = np.append(R_predict, p_lab.cpu().numpy())
                R_labels = np.append(R_labels, y)

                count += len(y)
                correct += torch.sum(p_lab == y.to(self.device)).item()
                di_correct += torch.sum(di_lab == y.to(self.device)).item()


            R_di_accuracy = di_correct / count * 100
            R_accuracy = correct / count * 100

            # IR test
            di_correct = 0
            correct = 0
            count, count_ir = 0, 0
            for x, y in ir_test_loader:

                x = x.to(self.device)
                x = self.nettarget_mapping(x)

                f_ci, f_di = self.netG_D(x), self.netG_T(x)
                if ir_task == 'rt' or ir_task == 'rs':
                    di_pred = self.netC_R(f_di)
                elif ir_task == 'irt' or ir_task == 'irs':
                    di_pred = self.netC_IR(f_di)
                # di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y).item()

                feat = self.netFS(f_ci, f_di)

                if ir_task == 'rt' or ir_task == 'rs':
                    pred = self.netC_R(feat)
                elif ir_task == 'irt' or ir_task == 'irs':
                    pred = self.netC_IR(feat)
                # correct += torch.sum(torch.argmax(pred, dim=1) == y).item()
                # count += len(y)

                p_lab = torch.argmax(pred, dim=1)
                di_lab = torch.argmax(di_pred, dim=1)
                # for la_i in range(len(self.irt_list)):
                #     p_lab[p_lab == la_i] = self.irt_list[la_i]
                #     di_lab[di_lab == la_i] = self.irt_list[la_i]
                ir_index_list, ir_di_index_list = [], []
                for la_i in range(self.irt_class):
                    ind_i = torch.where(p_lab == la_i)[0]
                    ir_index_list.append(ind_i)
                    di_ind_i = torch.where(di_lab == la_i)[0]
                    ir_di_index_list.append(di_ind_i)
                for la_i in range(len(self.irt_list)):
                    ind_i = ir_index_list[la_i]
                    di_ind_i = ir_di_index_list[la_i]
                    p_lab[ind_i] = self.irt_list[la_i]
                    di_lab[di_ind_i] = self.irt_list[la_i]
                # correct += torch.sum(p_lab == y).item()
                # count += len(y)

                IR_predict = np.append(IR_predict, p_lab.cpu().numpy())
                IR_labels = np.append(IR_labels, y)
                di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y.to(self.device)).item()
                correct += torch.sum(p_lab == y.to(self.device)).item()
                count_ir += len(y)


            IR_di_accuracy = di_correct / count_ir * 100
            IR_accuracy = correct / count_ir * 100


        for model in models:
            model.train()

        return R_accuracy, R_di_accuracy, IR_accuracy, IR_di_accuracy, R_predict, R_labels, IR_predict, IR_labels, self.rt_s2t_list#, re_fea.numpy()

    def getRTfeas(self, ft_iter, aug1_iter, aug2_iter, SELF_s2t=None, ITER=0):
        ft_iter = iter(ft_iter)
        aug1_iter = iter(aug1_iter)
        aug2_iter = iter(aug2_iter)

        models = [self.netG_D, self.netG_T, self.netFS, self.netC_R]
        for model in models:
            model.eval()

        di_correct = 0
        correct = 0
        count, count_ir = 0, 0

        R_predict = np.array([], dtype=np.int64)
        R_labels = np.array([], dtype=np.int64)

        fi_fea = torch.tensor(np.array([], dtype=np.int64))
        aug1_fea = torch.tensor(np.array([], dtype=np.int64))
        aug2_fea = torch.tensor(np.array([], dtype=np.int64))

        # ALL_predict = np.array([], dtype=np.int64)
        # ALL_labels = np.array([], dtype=np.int64)

        if SELF_s2t is None:
            print('Error*******************************')
        else:
            self.rt_s2t_list = SELF_s2t
        print('rt_s2t_list: ', self.rt_s2t_list)
        flag = 0
        # ITER = math.ceil(3297 / 128)
        with torch.no_grad():
            for iteration in range(ITER):
                im, lab, row_bat, clo_bat = next(ft_iter)
                im_q, _ = next(aug1_iter)
                im_k, _ = next(aug2_iter)
                im = im.to(self.device)
                im_q = im_q.to(self.device)
                im_k = im_k.to(self.device)
                im = self.nettarget_mapping(im)
                im_q = self.nettarget_mapping(im_q)
                im_k = self.nettarget_mapping(im_k)

                f_ci, f_di = self.netG_D(im), self.netG_T(im)
                aug1_ci, aug1_di = self.netG_D(im_q), self.netG_T(im_q)
                aug2_ci, aug2_di = self.netG_D(im_k), self.netG_T(im_k)

                f_feat = self.netFS(f_ci, f_di)
                aug1_feat = self.netFS(aug1_ci, aug1_di)
                aug2_feat = self.netFS(aug2_ci, aug2_di)

                f_pred = self.netC_R(f_feat)
                aug1_pred = self.netC_R(aug1_feat)
                aug2_pred = self.netC_R(aug2_feat)

                if flag == 0:
                    fi_fea = f_feat.squeeze()#.cpu().numpy()
                    aug1_fea = aug1_feat.squeeze()#.cpu().numpy()
                    aug2_fea = aug2_feat.squeeze()#.cpu().numpy()
                    flag = 1
                else:
                    fi_fea = torch.cat([fi_fea, f_feat.squeeze()], dim=0)
                    aug1_fea = torch.cat([aug1_fea, aug1_feat.squeeze()], dim=0)
                    aug2_fea = torch.cat([aug2_fea, aug2_feat.squeeze()], dim=0)

                f_p_lab = torch.argmax(f_pred, dim=1)
                aug1_p_lab = torch.argmax(aug1_pred, dim=1)
                aug2_p_lab = torch.argmax(aug2_pred, dim=1)

                # source to target
                index_list, di_index_list = [], []
                for la_i in range(self.rt_class):
                    ind_i = torch.where(f_p_lab == la_i)[0]
                    index_list.append(ind_i)

                for la_i in range(len(self.rt_list)):
                    ind_i = index_list[la_i]
                    f_p_lab[ind_i] = int(self.rt_s2t_list[la_i])



                index_list, di_index_list = [], []
                for la_i in range(self.rt_class):
                    ind_i = torch.where(f_p_lab == la_i)[0]
                    index_list.append(ind_i)
                for la_i in range(len(self.rt_list)):
                    ind_i = index_list[la_i]
                    f_p_lab[ind_i] = self.rt_list[la_i]

                R_predict = np.append(R_predict, f_p_lab.cpu().numpy())
                # R_labels = np.append(R_labels, lab)

                count += len(lab)
                correct += torch.sum(f_p_lab == lab.to(self.device)).item()

            R_accuracy = correct / count * 100
            print('R_accuracy:', R_accuracy)

        for model in models:
            model.train()

        fi_fea = fi_fea.cpu().numpy()
        aug1_fea = aug1_fea.cpu().numpy()
        aug2_fea = aug2_fea.cpu().numpy()

        return fi_fea, aug1_fea, aug2_fea, R_predict


    def graph_init(self, fea, fea_aug1, fea_aug2, depth=5, in_cha=128, out_cha=64):
        self.GCN_layers = depth
        self.netSAGE = GraphSage(depth, in_cha, out_cha, fea, self.adj_lists, self.device, raw_agu1_features=fea_aug1, raw_aug2_features=fea_aug2)
        self.netSAGE.to(self.device)
        nodes = np.arange(fea.shape[0])
        self.unsupervised_loss = UnsupervisedLoss(self.adj_lists, nodes, self.device)

        params = []
        for param in self.netSAGE.parameters():
            if param.requires_grad:
                params.append(param)
        self.optSAGE = torch.optim.Adam(params, lr=0.01)

    # def finetune(self, graph_data, im_w, im_q, im_k, lab, row_bat, clo_bat, test_loader, rt_s_data, rt_s_lab, SAVE=False,
    #              Epoch=1, Iter=0):
    def finetune(self, graph_data, Epoch=1, Iter=0, randomstate=234):

        fea = torch.tensor(graph_data['fea']).to(self.device)
        fea_aug1 = torch.tensor(graph_data['fea_aug1']).to(self.device)
        fea_aug2 = torch.tensor(graph_data['fea_aug2']).to(self.device)
        gt = torch.tensor(graph_data['gt'])
        no_lab = torch.tensor(graph_data['no_lab'])
        nodes = np.arange(gt.shape[0])
        b_sz = 64
        batches = math.ceil(len(gt) / b_sz)

        best_acc = 0
        best_aa = 0
        best_ka = 0
        best_ma = []
        best_epoch = 0
        best_epo_index = 0
        best_aalist = []

        for epoch in range(Epoch):
            train_nodes = nodes
            train_nodes = shuffle(train_nodes, random_state=randomstate) # 234
            visited_nodes = set()
            # model initial
            # models = [self.netSAGE]
            # for model in models:
            #     model.train()
            # updated_models = ['SAGE']
            # opts = self._get_opts(updated_models)
            # for opt in opts:
            #     opt.zero_grad()
            self.netSAGE.zero_grad()
            self.optSAGE.zero_grad()

            for epo_index in range(20): #batches):
                nodes_batch = train_nodes[epo_index * b_sz:(epo_index + 1) * b_sz]
                nodes_batch = np.asarray(list(self.unsupervised_loss.extend_nodes(nodes_batch, num_neg=100)))
                visited_nodes |= set(nodes_batch)

                im_w = fea[nodes_batch, :]#.to(self.device)
                im_q = fea_aug1[nodes_batch, :]#.to(self.device)
                im_k = fea_aug2[nodes_batch, :]#.to(self.device)
                lab = no_lab[nodes_batch]
                gt_b = gt[nodes_batch]

                # row_bat = row_bat[np.newaxis, :]
                # clo_bat = clo_bat[np.newaxis, :]
                # mix_row = np.concatenate([row_bat, row_bat, row_bat], axis=1)
                # mix_clo = np.concatenate([clo_bat, clo_bat, clo_bat], axis=1)
                # pos_emb = get_2d_sincos_pos_embed_from_grid(embed_dim=128,
                #                                             grid=np.concatenate([row_bat, clo_bat], axis=0))
                # mix_pos_emb = get_2d_sincos_pos_embed_from_grid(embed_dim=128,
                #                                                 grid=np.concatenate([mix_row, mix_clo], axis=0))
                # pos_emb = torch.tensor(pos_emb, dtype=torch.float32).to(self.device)
                # mix_pos_emb = torch.tensor(mix_pos_emb, dtype=torch.float32).to(self.device)
                # all_pos_emb = torch.cat([pos_emb, pos_emb, pos_emb, mix_pos_emb]).to(self.device)
                MIX_seed = np.random.randint(10, 9999, size=1)[0]
                im_mix, mix_randind, mix_lam = mixup(torch.cat([im_w, im_q, im_k]), alpha=1.0, seed=MIX_seed)
                # all_gt_lab = lab.repeat(1, 6).squeeze()
                # mix_gt_lab = lab.repeat(1, 3).squeeze()
                # mix_gt_lab = torch.index_select(mix_gt_lab, 0, mix_randind)
                # all_gt_lab[im_mix.shape[0]:] = mix_gt_lab

                x = torch.cat([im_w, im_q, im_k, im_mix])
                # x = x.to(self.device)
                # x = self.nettarget_mapping(x)
                x_q = x  # + all_pos_emb
                q_logits = self.netC_R(x_q)

                # q_pro = self.netprojector(x_q.clone())
                # graph SAGE process
                q_pro_w = self.netSAGE(nodes_batch)
                q_pro_q = self.netSAGE.forward_aug1(nodes_batch)
                q_pro_k = self.netSAGE.forward_aug2(nodes_batch)

                # cluster
                train_features = q_pro_w.detach().cpu().numpy()
                if epoch == 0 and epo_index == 0:
                    print('DIAGNOSTIC - GraphSAGE output stats:')
                    print('q_pro_w std:', train_features.std(), 'mean:', train_features.mean(), 'shape:', train_features.shape)
                    print('q_pro_w unique rows (rounded):', len(set(map(tuple, train_features.round(3)[:200]))), 'out of', min(200, train_features.shape[0]), 'checked')
                if np.isnan(train_features).any():
                    train_features = np.nan_to_num(train_features)
                train_features_q = q_pro_q.detach().cpu().numpy()
                if np.isnan(train_features_q).any():
                    train_features_q = np.nan_to_num(train_features_q)
                train_features_k = q_pro_k.detach().cpu().numpy()
                if np.isnan(train_features_k).any():
                    train_features_k = np.nan_to_num(train_features_k)

                # Km_model = KMeans(init='random', n_clusters=self.rt_class)
                # Km_model.fit(q_pro_w.detach().cpu().numpy())
                # from sklearn_extra.cluster import KMedoids, CLARA, CommonNNClustering  # unused, and incompatible with numpy 2.x anyway
                from sklearn.cluster import DBSCAN
                # Km_model = KMedoids(init='k-medoids++', n_clusters=self.rt_class)#, random_state=MIX_seed)
                Km_model = KMeans(init='random', n_clusters=self.rt_class, max_iter=1000)
                # Km_model = CommonNNClustering(eps=0.5, min_samples=5)

                # Km_model = CLARA(n_clusters=self.rt_class, max_iter=1000, random_state=MIX_seed+1)
                Km_model.fit(train_features)
##########################################################
                km_center = Km_model.cluster_centers_
                q_logits_w = self.genlogits(train_features, km_center)
                q_logits_q = self.genlogits(train_features_q, km_center)
                q_logits_k = self.genlogits(train_features_k, km_center)

                prad_lab = Km_model.predict(train_features)
                prad_lab_q = Km_model.predict(train_features_q)
                prad_lab_k = Km_model.predict(train_features_k)

                C_r = metrics.confusion_matrix(gt_b, prad_lab)
                ######## label change
                Changelist = self.get_changelist(C_r)
                # print('change list:', Changelist)
                index_store = []
                class_num = self.rt_class
                for ind_i in range(class_num):
                    ind_cla = np.where(prad_lab == ind_i)[0]
                    index_store.append(ind_cla)
                for ind_i in range(class_num):
                    ind_cla = index_store[Changelist[ind_i]]
                    prad_lab[ind_cla] = ind_i
                C_r = metrics.confusion_matrix(gt_b, prad_lab)
                #################
                AA = np.diag(C_r) / np.sum(C_r, 1, dtype=float)
                AA_r = np.mean(AA, 0)

                correct = np.sum(gt_b.numpy() == prad_lab)
                acc = correct / len(gt_b) * 100
                # print('acc:', acc)
                kappa = metrics.cohen_kappa_score(gt_b, prad_lab)

                # model save
                if AA_r >= best_aa:
                    best_acc = acc
                    best_aa = AA_r
                    best_ma = C_r
                    best_epoch = epoch
                    best_aalist = AA
                    best_epo_index = epo_index
                    best_ka = kappa
                    print('best acc:', epoch, epo_index, best_acc)
                    # save model
                    pickle.dump(Km_model, open(self.save_dir+'/best_km_'+str(Iter)+'.pkl', 'wb'))
                    torch.save(self.netSAGE.state_dict(), self.save_dir+'/best_GSAGE_'+str(Iter)+'.pth')
                print('DEBUG every-iter acc:', epoch, epo_index, acc)


                self.writer.add_scalar('acc/OA', acc, int(epoch * 26 + epo_index))
                self.writer.add_scalar('acc/AA', AA_r, int(epoch * 26 + epo_index))
                self.writer.add_scalar('acc/Ka', kappa, int(epoch * 26 + epo_index))

                graph_loss = self.unsupervised_loss.get_loss_sage(q_pro_w, nodes_batch)
#########################################################################################
                # q_pro_mix, mix_randind_q, mix_lam_q = mixup(torch.cat([q_pro_w, q_pro_q, q_pro_k]), alpha=1.0, seed=MIX_seed)
                q_pro_mix = torch.cat([q_pro_w, q_pro_q, q_pro_k])
                q_pro_mix = q_pro_mix[mix_randind]
                # q_log_mix, mix_randind_log, mix_lam_log = mixup(torch.cat([q_logits_w, q_logits_q, q_logits_k]), alpha=1.0, seed=MIX_seed)
                q_log_mix = torch.cat([q_logits_w, q_logits_q, q_logits_k])
                q_log_mix = q_log_mix[mix_randind]
                prad_lab = torch.tensor(prad_lab).to(self.device)
                prad_lab_q = torch.tensor(prad_lab_q).to(self.device)
                prad_lab_k = torch.tensor(prad_lab_k).to(self.device)
                # q_lab_mix, mix_randind_lab, mix_lam_lab = mixup(torch.cat([prad_lab, prad_lab_q, prad_lab_k]), alpha=1.0, seed=MIX_seed)
                q_lab_mix = torch.cat([prad_lab, prad_lab_q, prad_lab_k])
                q_lab_mix = q_lab_mix[mix_randind]
                self.pseudo_labels = torch.cat([prad_lab, prad_lab_q, prad_lab_k, q_lab_mix])
                self.pseudo_labels = self.label_project(self.pseudo_labels)

                contrastive_loss = self.loss(q_pro_q, q_pro_k)
                confidences = torch.zeros([self.pseudo_labels.shape[0], ])
                cls_loss1, cls_loss2, align_loss = self.forward_cls_loss(gt_b, confidences, q_pro_w,
                                                                         q_logits_w,
                                                                         q_logits_q,
                                                                         q_logits_k,
                                                                         q_log_mix,
                                                                         q_pro_mix,
                                                                         mix_randind,
                                                                         mix_lam,
                                                                         )
                # forward_reg_loss
                ent_loss, ne_loss = self.forward_reg_loss(torch.cat([q_logits_q, q_logits_k, q_log_mix]))

                loss = 10.0 * graph_loss + contrastive_loss + 0.5 * (ent_loss + ne_loss)
                       # matches paper's Eq.12: Ltotal = lambda1*LG + Lctr + lambda2*Lreg, lambda1=10, lambda2=0.5
                       # 1.0 * (cls_loss1 + cls_loss2) + \
                       # 1.0 * align_loss #+\

                       # 10.0 * ne_loss # + \

                self.writer.add_scalar('loss/contrastive', contrastive_loss, int(epoch * 26 + epo_index))
                self.writer.add_scalar('loss/ent_loss', ent_loss, int(epoch * 26 + epo_index))
                self.writer.add_scalar('loss/ne_loss', ne_loss, int(epoch * 26 + epo_index))
                self.writer.add_scalar('loss/cls_loss1', cls_loss1, int(epoch * 26 + epo_index))
                self.writer.add_scalar('loss/cls_loss2', cls_loss2, int(epoch * 26 + epo_index))
                self.writer.add_scalar('loss/align_loss', align_loss, int(epoch * 26 + epo_index))
                self.writer.add_scalar('loss/graph_loss', graph_loss, int(epoch * 26 + epo_index))

                # loss1 = loss.requires_grad_(True)
                # loss1.backward(retain_graph=True)
                # for opt in opts:
                #     opt.step()
                loss.backward()
                nn.utils.clip_grad_norm_(self.netSAGE.parameters(), 5)
                self.optSAGE.step()
                self.optSAGE.zero_grad()
                self.netSAGE.zero_grad()


#################### test and save load
        print('best acc:', best_acc)
        print('best AA:', best_aa)
        print(best_aalist)
        print(best_ma)
        print(best_epoch, best_epo_index)

        print('test---------------------------------------')
        model = GraphSage(self.GCN_layers, 128, 64, fea, self.adj_lists, self.device)
        model.load_state_dict(torch.load(self.save_dir+'/best_GSAGE_'+str(Iter)+'.pth'))
        kmmodel = pickle.load(open(self.save_dir+'/best_km_'+str(Iter)+'.pkl', "rb"))
        model.to(self.device)
        model.eval()
        embs_batch = model(train_nodes)
        test_embs = embs_batch.detach().cpu().numpy()
        if np.isnan(test_embs).any():
            test_embs = np.nan_to_num(test_embs)
        prad_test = kmmodel.predict(test_embs)
        gt_test = gt[train_nodes]
        C_r = metrics.confusion_matrix(gt_test, prad_test)
        # print('ori C_r:')
        # print(C_r)
        Changelist = self.get_changelist(C_r)
        # print('change list:', Changelist)
        index_store = []
        class_num = self.rt_class
        for ind_i in range(class_num):
            ind_cla = np.where(prad_test == ind_i)[0]
            index_store.append(ind_cla)
        for ind_i in range(class_num):
            ind_cla = index_store[Changelist[ind_i]]
            prad_test[ind_cla] = ind_i
        C_r = metrics.confusion_matrix(gt_test, prad_test)
        # print('new C-r:')
        # print(C_r)
        AA = np.diag(C_r) / np.sum(C_r, 1, dtype=float)
        AA_r = np.mean(AA, 0)
        # print('AA:', AA_r, AA)
        correct = np.sum(gt_test.numpy() == prad_test)
        acc = correct / len(gt_test) * 100
        # print('acc:', acc)
        kappa = metrics.cohen_kappa_score(gt_test, prad_test)
        # print('kappa:', kappa)

        # gen output
        out_lab = self.project2RT(torch.tensor(prad_test)).numpy()
        fin_lab = np.zeros_like(out_lab)
        for i in range(len(train_nodes)):
            fin_lab[train_nodes[i]] = out_lab[i]

        return acc, AA_r, AA, fin_lab, best_epoch, best_epo_index


    def genlogits(self, data, center):
        dist_ma = np.zeros([data.shape[0], center.shape[0]])
        for i in range(center.shape[0]):
            cen_i = center[i, ][np.newaxis, :]
            # print(cen_i.shape[0])
            dist = np.linalg.norm((data - cen_i), axis=1)
            # print(dist.shape)
            dist_ma[:, i] = dist
        dist_ma = (dist_ma - dist_ma.min()) / dist_ma.max()
        logits = np.exp(-dist_ma)
        logits = torch.tensor(logits).to(self.device)
        return logits

    def matrix_zero(self, ma, row, clo):
        ma[row, :] = 0
        ma[:, clo] = 0
        return ma

    def get_changelist(self, C_r):
        array_1d = C_r.reshape([self.rt_class ** 2, ])
        # print(array_1d)
        flag = 0
        Changelist = []
        for last_i in range(self.rt_class):
            Changelist.append(-1)
        while flag == 0:
            array_1d = C_r.reshape([self.rt_class ** 2, ])
            # print(array_1d)
            num_max = np.max(array_1d)
            if num_max == 0:
                flag = 1
                break
            else:
                idx = np.argsort(array_1d)
                max_idx = idx[-1]
                row = max_idx // self.rt_class
                clo = max_idx % self.rt_class
                # print(row, clo)
                C_r = self.matrix_zero(C_r, row, clo)
                # print(C_r)
                Changelist[row] = clo
        #   zero-process
        # get rest
        rest_gt = np.where(np.array(Changelist) == -1)[0]
        rest_gt_num = len(rest_gt)
        # class_list = [-1, 0, 1, 2, 3, 4, 5]
        class_list = []
        for cla_i in range(self.rt_class+1):
            class_list.append(cla_i-1)
        rest_cla = list(set(class_list) - set(Changelist))
        # print('rest class:', rest_cla)
        if -1 in rest_cla:
            index = np.where(np.array(rest_cla) == -1)[0]
            if len(index) == 1:
                rest_cla.pop(index[0])
            else:
                print('Error*********************************')
        rest_cla_num = len(rest_cla)
        # print(rest_cla)
        if rest_cla_num != 0 and (rest_gt_num == rest_gt_num):
            for i in range(rest_cla_num):
                Changelist[rest_gt[i]] = rest_cla[i]
        else:
            print('*********************************')
        print(Changelist)
        return Changelist

    def fintuning_save(self, test_features, features, clean_labels):
        # data save
        save_path = self.save_dir
        mat_name = 'stage2_%s.mat' % (str(self.best_rt_acc))
        sio.savemat(os.path.join(save_path, mat_name),
                    {'test_features': test_features.cpu().detach().numpy(),
                     'features': features.cpu().detach().numpy(),
                     'clean_labels': clean_labels.cpu().detach().numpy()})
        # model save
        model_names = ['G_D', 'G_T', 'D_D', 'FS', 'C_R', 'C_IR', 'projector']
        for name in model_names:
            if isinstance(name, str):
                save_filename = 'stage2_%s_%s.pth' % (str(self.best_rt_acc), name)
                save_dir = os.path.join(save_path, save_filename)
                net = getattr(self, 'net' + name)
                if self.device != torch.device('cpu'):
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_dir)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_dir)

    def fintuning_saveload(self, best_rt_acc):
        # data load
        save_path = self.save_dir
        mat_name = 'stage2_%s.mat' % (best_rt_acc)
        data = sio.loadmat(os.path.join(save_path, mat_name))
        self.fintuning_test_features = data['test_features']
        self.fintuning_features = data['features']
        self.fintuning_clean_labels = data['clean_labels']


        # model load
        model_names = ['G_D', 'G_T', 'D_D', 'FS', 'C_R', 'C_IR', 'projector']
        for name in self.model_names:
            if name == 'D_D':
                continue
            load_filename = 'stage2_%s_%s.pth' % (best_rt_acc, name)
            load_path = os.path.join(save_path, load_filename)
            net = getattr(self, 'net' + name)

            if not os.path.exists(load_path):
                print(load_path, "not exists")
                continue

            # if isinstance(net, torch.nn.DataParallel):
            #     net = net.module
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

    def fintuning_test(self, test_loader):
        test_features, test_cluster_labels, test_labels = self.extract_features(test_loader)
        test_pred_labels = torch.argmax(test_cluster_labels, dim=1)
        test_pred_labels = self.label_project(test_pred_labels)
        test_acc = (test_labels == test_pred_labels).float().mean()
        print('test_acc ', test_acc)

        from .utils.knn_monitor import knn_predict
        # self.fintuning_test_features = data['test_features']
        # self.fintuning_features = data['features']
        # self.fintuning_clean_labels = data['clean_labels']
        knn_labels = knn_predict(self.fintuning_test_features, self.fintuning_features, self.fintuning_clean_labels,  # rt_s_lab
                                 classes=self.rt_class, knn_k=100, knn_t=0.1)[:, 0]
        knn_acc = (test_labels == knn_labels).float().mean().cpu().detach().numpy()
        print('knn_acc ', knn_acc)

        return test_acc, test_pred_labels.cpu().detach().numpy(), knn_acc, knn_labels.cpu().detach().numpy()

    def extract_features(self, testloader):
        models = [self.netG_D, self.netG_T, self.netFS, self.netC_R, self.netC_IR, self.netprojector]
        for model in models:
            model.eval()

        res_pred = torch.tensor(np.array([], dtype=np.int64))
        gt_labs = torch.tensor(np.array([], dtype=np.int64))
        pre_fea = torch.tensor(np.array([], dtype=np.int64))


        flag = 0
        for x, y, row, clo in testloader:
            row_bat = row[np.newaxis, :]
            clo_bat = clo[np.newaxis, :]
            pos_emb = get_2d_sincos_pos_embed_from_grid(embed_dim=128, grid=np.concatenate([row_bat, clo_bat], axis=0))
            pos_emb = torch.tensor(pos_emb, dtype=torch.float32).to(self.device)

            x = x.to(self.device)
            x = self.nettarget_mapping(x)
            f_ci, f_di = self.netG_D(x), self.netG_T(x)
            feat = self.netFS(f_ci, f_di).squeeze() + pos_emb

            p_pro = self.netprojector(feat)
            pred = self.netC_R(feat)
            # p_lab = torch.argmax(pred, dim=1)

            if flag == 0:
                res_pred = pred#.cpu().detach().numpy()
                gt_labs = y#.cpu().detach().numpy()
                pre_fea = p_pro#.cpu().detach().numpy()
                flag = 1
            else:
                # res_pred = np.concatenate((res_pred, pred.cpu().detach().numpy()), axis=0)
                # gt_labs = np.concatenate((gt_labs, y.cpu().detach().numpy()), axis=0)
                # pre_fea = np.concatenate((pre_fea, p_pro.cpu().detach().numpy()), axis=0)
                res_pred = torch.cat([res_pred, pred], dim=0)
                gt_labs = torch.cat([gt_labs, y], dim=0)
                pre_fea = torch.cat([pre_fea, p_pro], dim=0)

        # label change
        r_index = []
        for in_lab in range(len(self.rt_list)):
            ind_i = torch.where(gt_labs == self.rt_list[in_lab])[0]
            r_index.append(ind_i)
        for in_lab in range(len(self.rt_list)):
            ind_i = r_index[in_lab]
            gt_labs[ind_i] = in_lab

        for model in [self.netC_R, self.netprojector]:
            model.train()

        return pre_fea, res_pred, gt_labs.to(self.device)

    def label_project(self, p_lab):
        index_list, di_index_list = [], []
        for la_i in range(self.rt_class):
            ind_i = torch.where(p_lab == la_i)[0]
            index_list.append(ind_i)

        for la_i in range(len(self.rt_list)):
            ind_i = index_list[la_i]
            p_lab[ind_i] = int(self.rt_s2t_list[la_i])
        return p_lab

    def project2RT(self, p_lab):
        index_list, di_index_list = [], []
        for la_i in range(self.rt_class):
            ind_i = torch.where(p_lab == la_i)[0]
            index_list.append(ind_i)

        for la_i in range(len(self.rt_list)):
            ind_i = index_list[la_i]
            p_lab[ind_i] = int(self.rt_list[la_i])
        return p_lab

    def forward_cls_loss(self,
                         pseudo_labels, confidences, q_w, w_logits,
                         q_logits1, q_logits2, mix_logits,
                         q_mix, mix_randind, mix_lam):
#############################################
        # with torch.no_grad():
        labels = pseudo_labels.to(self.device)
        confidences = confidences.unsqueeze(1)[0:q_w.shape[0]].to(self.device)

        targets_onehot_noise = F.one_hot(labels, self.num_cluster).float().cuda()
        w_prob = F.softmax(w_logits.detach(), dim=1)
        q_prob1 = F.softmax(q_logits1.detach(), dim=1)
        q_prob2 = F.softmax(q_logits2.detach(), dim=1)

        # targets_mix_corrected = (w_prob + q_prob1 + q_prob2) / 3.

        def comb(p1, p2, lam):
            return (1 - lam) * p1 + lam * p2

        mix_lam = mix_lam.to(self.device)
        targets_corrected1 = comb(q_prob2, targets_onehot_noise, confidences * self.scale1)
        targets_corrected2 = comb(q_prob1, targets_onehot_noise, confidences * self.scale1)
        targets_mix_corrected = comb((q_prob1 + q_prob2) * 0.5, targets_onehot_noise, confidences * self.scale2)
        targets_mix_corrected = targets_mix_corrected.repeat((q_mix.size(0) // q_logits1.size(0), 1))
        targets_mix_corrected = comb(targets_mix_corrected[mix_randind], targets_mix_corrected, mix_lam)

        targets_mix_noise = targets_onehot_noise.repeat((q_mix.size(0) // q_logits1.size(0), 1))
        targets_mix_noise = comb(targets_mix_noise[mix_randind], targets_mix_noise, mix_lam)
#######################################
        proto = self.prototypes.detach().T.to(self.device)
        align_logits = q_mix.mm(proto) / self.T

        def CE(logits, targets):
            return - (targets * F.log_softmax(logits, dim=1)).sum(-1).mean()

        if self.warmup:
            cls_loss1 = F.cross_entropy(q_logits1, labels) + \
                        F.cross_entropy(q_logits2, labels)
            cls_loss2 = CE(mix_logits, targets_mix_noise)
            align_loss = CE(align_logits, targets_mix_noise)
        else:
            # align_loss_mix = targets_mix_corrected
            align_loss = CE(align_logits, targets_mix_corrected)
            cls_loss1 = CE(q_logits1, targets_corrected1) + \
                        CE(q_logits2, targets_corrected2)
            cls_loss2 = CE(mix_logits, targets_mix_corrected)

        return cls_loss1, cls_loss2, align_loss

    def forward_reg_loss(self, pred_logits):
        pred_softmax = F.softmax(pred_logits, dim=1)
        ent_loss = - (pred_softmax * F.log_softmax(pred_logits, dim=1)).sum(dim=1).mean()
        prob_mean = pred_softmax.mean(dim=0)
        ne_loss = (prob_mean * prob_mean.log()).sum()
        return ent_loss, ne_loss


    def get_current_loss(self):
        losses = {}
        for k in self.losses:
            losses[k] = self.losses[k] / self.counts[k]
        return losses

    def save_networks(self, epoch, iter):
        for name in self.model_names:
            if isinstance(name, str):
                # save_filename = 'stage1%s_%s__%s.pth' % (iter, epoch, name)
                save_filename = 'stage1%s__%s.pth' % (iter, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.device != torch.device('cpu'):
                    # torch.save(net.module.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)


    def load_networks(self, iter, path=None, s2t=None):
        for name in self.model_names:
            if name == 'D_D':
                continue
            # elif name == 'projector':
            #     load_filename = '%s_net_%s.pth' % (epoch, 'C_R')
            else:
                load_filename = 'stage1%s__%s.pth' % (iter, name)
            if path is None:
                load_path = os.path.join(self.save_dir, load_filename)
            else:
                load_path = os.path.join(path, load_filename)

            net = getattr(self, 'net' + name)

            if not os.path.exists(load_path):
                print(load_path, "not exists")
                continue

            # if isinstance(net, torch.nn.DataParallel):
            #     net = net.module
            state_dict = torch.load(load_path, map_location=str(self.device))
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=False for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            if self.lr_policy == 'plateau':
                pass
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test_withoupos(self, r_test_loader, r_task, ir_test_loader, ir_task, rt_s_data, rt_s_lab, ):
        models = [self.netG_D, self.netG_T, self.netFS, self.netC_R, self.netC_IR, self.netprojector]
        for model in models:
            model.eval()

        di_correct = 0
        correct = 0
        count, count_ir = 0, 0

        IR_predict = np.array([], dtype=np.int64)
        IR_labels = np.array([], dtype=np.int64)
        R_predict = np.array([], dtype=np.int64)
        R_labels = np.array([], dtype=np.int64)

        re_fea = torch.tensor(np.array([], dtype=np.int64))
        re_data = torch.tensor(np.array([], dtype=np.int64))

        # ALL_predict = np.array([], dtype=np.int64)
        # ALL_labels = np.array([], dtype=np.int64)

        ##### save rt_s mean features
        print('rt_s')
        for i in range(self.rt_class):
            index = torch.where(rt_s_lab == i)[0]
            da = rt_s_data[index, :]

            cla_name = self.rt_list[i]
            index_t = torch.where(self.T_r_labels == torch.tensor(cla_name).to(self.device))[0]
            da_t = self.T_r_data[index_t, ]
            with torch.no_grad():
                da = da.to(self.device)
                da = self.netsource_mapping(da)
                f_ci, f_di = self.netG_D(da), self.netG_T(da)
                feat = self.netFS(f_ci, f_di)

                da_t = da_t.to(self.device)
                da_t = self.nettarget_mapping(da_t)
                f_ci_t, f_di_t = self.netG_D(da_t), self.netG_T(da_t)
                feat_t = self.netFS(f_ci_t, f_di_t)
            mean = torch.squeeze(torch.mean(feat, dim=0))
            self.rt_s_proto[i, ] = mean

            mean_t = torch.squeeze(torch.mean(feat_t, dim=0))
            self.rt_t_proto[i, ] = mean_t

        # simalarity measure
        for i in range(self.rt_class):
            sim = torch.cosine_similarity(torch.unsqueeze(self.rt_s_proto[i, ], dim=0), self.rt_t_proto, dim=1)
            print(sim)
            self.sim_matrix[:, i] = sim
            max_index = torch.argmax(sim)
            # self.rt_s2t_list[i] = max_index
        print('rt_s2t_list: ', self.rt_s2t_list)
        print('sim matrix: ')
        print(self.sim_matrix)
        s2t = torch.zeros([self.rt_class, ])
        s2t = s2t - 1
        ma = self.sim_matrix.reshape([self.rt_class ** 2, 1])
        max_index = torch.argsort(ma, dim=0, descending=True)
        cnt = 0
        i = 0
        while cnt < self.rt_class:
            id = max_index[i]
            row = id // self.rt_class
            clo = id % self.rt_class
            if s2t[clo] == -1:
                if row not in s2t:
                    s2t[clo] = int(row)
                    cnt += 1
                    i += 1
                else:
                    i += 1
            else:
                i += 1
        self.rt_s2t_list = s2t
        print('rt_s2t_list: ', self.rt_s2t_list)

        #####
        # R-task
        flag = 0
        with torch.no_grad():
            for x, y, in r_test_loader:
                x = x.to(self.device)
                # if flag == 0:
                #     re_data = x[:, :, 4, 4].cpu()
                # else:
                #     re_data = torch.cat([re_data, x[:, :, 4, 4].cpu()], dim=0)
                x = self.nettarget_mapping(x)

                f_ci, f_di = self.netG_D(x), self.netG_T(x)
                if r_task == 'rt' or r_task == 'rs':
                    di_pred = self.netC_R(f_di)
                elif r_task == 'irt' or r_task == 'irs':
                    di_pred = self.netC_IR(f_di)

                # di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y).item()
                di_lab = torch.argmax(di_pred, dim=1)
                # for la_i in range(len(self.rt_list)):
                #     di_lab[di_lab == la_i] = self.rt_list[la_i]

                feat = self.netFS(f_ci, f_di)
                if r_task == 'rt' or r_task == 'rs':
                    pred = self.netC_R(feat)
                elif r_task == 'irt' or r_task == 'irs':
                    pred = self.netC_IR(feat)


                # if flag == 0:
                #     re_fea = feat.squeeze().cpu()
                #     flag =1
                # else:
                #     re_fea = torch.cat([re_fea, feat.squeeze().cpu()], dim=0)

                p_lab = torch.argmax(pred, dim=1)
                # for la_i in range(len(self.rt_list)):
                #     p_lab[p_lab == la_i] = self.rt_list[la_i]
                #     di_lab[di_lab == la_i] = self.rt_list[la_i]

                # source to target
                index_list, di_index_list = [], []
                for la_i in range(self.rt_class):
                    ind_i = torch.where(p_lab == la_i)[0]
                    index_list.append(ind_i)
                    di_ind_i = torch.where(di_lab == la_i)[0]
                    di_index_list.append(di_ind_i)
                for la_i in range(len(self.rt_list)):
                    ind_i = index_list[la_i]
                    di_ind_i = di_index_list[la_i]
                    p_lab[ind_i] = int(self.rt_s2t_list[la_i])
                    di_lab[di_ind_i] = int(self.rt_s2t_list[la_i])



                index_list, di_index_list = [], []
                for la_i in range(self.rt_class):
                    ind_i = torch.where(p_lab == la_i)[0]
                    index_list.append(ind_i)
                    di_ind_i = torch.where(di_lab == la_i)[0]
                    di_index_list.append(di_ind_i)
                for la_i in range(len(self.rt_list)):
                    ind_i = index_list[la_i]
                    di_ind_i = di_index_list[la_i]
                    p_lab[ind_i] = self.rt_list[la_i]
                    di_lab[di_ind_i] = self.rt_list[la_i]

                ##########################################################
                R_predict = np.append(R_predict, p_lab.cpu().numpy())
                R_labels = np.append(R_labels, y)

                count += len(y)
                correct += torch.sum(p_lab == y.to(self.device)).item()
                di_correct += torch.sum(di_lab == y.to(self.device)).item()


            R_di_accuracy = di_correct / count * 100
            R_accuracy = correct / count * 100

            # IR test
            for x, y in ir_test_loader:

                x = x.to(self.device)
                x = self.nettarget_mapping(x)

                f_ci, f_di = self.netG_D(x), self.netG_T(x)
                if ir_task == 'rt' or ir_task == 'rs':
                    di_pred = self.netC_R(f_di)
                elif ir_task == 'irt' or ir_task == 'irs':
                    di_pred = self.netC_IR(f_di)
                # di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y).item()

                feat = self.netFS(f_ci, f_di)

                if ir_task == 'rt' or ir_task == 'rs':
                    pred = self.netC_R(feat)
                elif ir_task == 'irt' or ir_task == 'irs':
                    pred = self.netC_IR(feat)
                # correct += torch.sum(torch.argmax(pred, dim=1) == y).item()
                # count += len(y)

                p_lab = torch.argmax(pred, dim=1)
                di_lab = torch.argmax(di_pred, dim=1)
                # for la_i in range(len(self.irt_list)):
                #     p_lab[p_lab == la_i] = self.irt_list[la_i]
                #     di_lab[di_lab == la_i] = self.irt_list[la_i]
                ir_index_list, ir_di_index_list = [], []
                for la_i in range(self.irt_class):
                    ind_i = torch.where(p_lab == la_i)[0]
                    ir_index_list.append(ind_i)
                    di_ind_i = torch.where(di_lab == la_i)[0]
                    ir_di_index_list.append(di_ind_i)
                for la_i in range(len(self.irt_list)):
                    ind_i = ir_index_list[la_i]
                    di_ind_i = ir_di_index_list[la_i]
                    p_lab[ind_i] = self.irt_list[la_i]
                    di_lab[di_ind_i] = self.irt_list[la_i]
                # correct += torch.sum(p_lab == y).item()
                # count += len(y)

                IR_predict = np.append(IR_predict, p_lab.cpu().numpy())
                IR_labels = np.append(IR_labels, y)
                di_correct += torch.sum(torch.argmax(di_pred, dim=1) == y.to(self.device)).item()
                correct += torch.sum(p_lab == y.to(self.device)).item()
                count_ir += len(y)


            IR_di_accuracy = di_correct / count_ir * 100
            IR_accuracy = correct / count_ir * 100


        for model in models:
            model.train()

        return R_accuracy, R_di_accuracy, IR_accuracy, IR_di_accuracy, R_predict, R_labels, IR_predict, IR_labels, #re_data.numpy(), re_fea.numpy()


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb    # batchsize * fea_dim


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def mixup(input, alpha=1.0, seed=123):
    bs = input.size(0)
    torch.manual_seed(seed)
    randind = torch.randperm(bs).to(input.device)  # 将0~n-1（包括0和n-1）随机打乱后获得的数字序列
    # beta = torch.distributions.beta.Beta(alpha, alpha)
    # lam = beta.sample([bs]).to(input.device)
    # import numpy as np
    np.random.seed(seed)
    lam = np.random.beta(alpha, alpha)  # beta 分布
    lam = torch.ones_like(randind).float() * lam
    lam = torch.max(lam, 1. - lam)
    lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
    input = lam_expanded * input + (1. - lam_expanded) * input[randind]
    return input, randind, lam.unsqueeze(1)

if __name__ == '__main__':


    mixup_alpha = 1.0
    im_w = torch.rand([9, 128, 128, 3])
    im_q = torch.rand([9, 128, 128, 3])
    im_k = torch.rand([9, 128, 128, 3])
    im_mix, mix_randind, mix_lam = mixup(torch.cat([im_w, im_q, im_k]), alpha=mixup_alpha)


    cfg = [[]]
    model = ZSDAModel(cfg)
    print('end')
    s_ir_data = torch.tensor(np.random.random([160, 128, 9, 9]), dtype=torch.float)
    s_ir_labels = torch.tensor(np.random.randint(low=0, high=8, size=[160, ]))
    s_r_data = torch.tensor(np.random.random([160, 128, 9, 9]), dtype=torch.float)
    s_r_labels = torch.tensor(np.random.randint(low=0, high=8, size=[160, ]))

    t_ir_data = torch.tensor(np.random.random([160, 200, 9, 9]), dtype=torch.float)
    t_ir_labels = torch.tensor(np.random.randint(low=0, high=8, size=[160, ]))
    t_r_data = torch.tensor(np.random.random([160, 200, 9, 9]), dtype=torch.float)
    t_r_labels = torch.tensor(np.random.randint(low=8, high=16, size=[160, ]))

    model.set_hyper_input(irt_s=s_ir_data, irt_s_lab=s_ir_labels, irt_t=t_ir_data, irt_t_lab=t_ir_labels,
                          rt_s=s_r_data, rt_s_lab=s_r_labels)
    losses = model.update()

    # losses = model.get_current_loss()
    model.update_learning_rate()

    accuracy, di_accuracy = model.test(t_r_data, t_r_labels)