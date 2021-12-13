import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image, ImageDraw, ImageOps
import torch.nn.utils.spectral_norm as spectral_norm
from models.preprocess import preprocess_item
if torch.cuda.is_available():
    device = torch.device('cuda:0')


def add_pool(x, nd_to_sample):
    dtype, device = x.dtype, x.device
    batch_size = torch.max(nd_to_sample) + 1
    pooled_x = torch.zeros(batch_size, x.shape[-1]).float().to(device)
    pool_to = nd_to_sample.view(-1, 1).expand_as(x).to(device)
    pooled_x = pooled_x.scatter_add(0, pool_to, x)
    return pooled_x


def compute_gradient_penalty(D, x, x_fake, given_y=None, given_w=None,
                             nd_to_sample=None, data_parallel=None,
                             ed_to_sample=None):
    indices = nd_to_sample, ed_to_sample
    batch_size = torch.max(nd_to_sample) + 1
    dtype, device = x.dtype, x.device
    u = torch.FloatTensor(x.shape[0], 1, 1).to(device)
    u.data.resize_(x.shape[0], 1, 1)
    u.uniform_(0, 1)
    x_both = x.data*u + x_fake.data*(1-u)
    x_both = x_both.to(device)
    x_both = Variable(x_both, requires_grad=True)
    grad_outputs = torch.ones(batch_size, 1).to(device)
    if data_parallel:
        _output = data_parallel(
            D, (x_both, given_y, given_w, nd_to_sample), indices)
    else:
        _output = D(x_both, given_y, given_w, nd_to_sample)
    grad = torch.autograd.grad(outputs=_output, inputs=x_both, grad_outputs=grad_outputs,
                               retain_graph=True, create_graph=True, only_inputs=True)[0]
    gradient_penalty = ((grad.norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    return gradient_penalty


def conv_block(in_channels, out_channels, k, s, p, act=None, upsample=False, spec_norm=False, batch_norm=False):
    block = []

    if upsample:
        if spec_norm:
            block.append(spectral_norm(torch.nn.ConvTranspose2d(in_channels, out_channels,
                                                                kernel_size=k, stride=s,
                                                                padding=p, bias=True)))
        else:
            block.append(torch.nn.ConvTranspose2d(in_channels, out_channels,
                                                  kernel_size=k, stride=s,
                                                  padding=p, bias=True))
    else:
        if spec_norm:
            block.append(spectral_norm(torch.nn.Conv2d(in_channels, out_channels,
                                                       kernel_size=k, stride=s,
                                                       padding=p, bias=True)))
        else:
            block.append(torch.nn.Conv2d(in_channels, out_channels,
                                         kernel_size=k, stride=s,
                                         padding=p, bias=True))
    if batch_norm:
        block.append(nn.BatchNorm2d(out_channels))
    if "leaky" in act:
        block.append(torch.nn.LeakyReLU(0.1, inplace=True))
    elif "relu" in act:
        block.append(torch.nn.ReLU(inplace=True))
    # elif "tanh":
    #     block.append(torch.nn.Tanh())
    return block


class Graphormer(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        hidden_dim,
        dropout_rate,
        intput_dropout_rate,
        weight_decay,
        ffn_dim,
        dataset_name,
        warmup_updates,
        tot_updates,
        peak_lr,
        end_lr,
        edge_type,
        multi_hop_max_dist,
        attention_dropout_rate,
        num_class,

    ):
        super().__init__()

        self.num_heads = num_heads
        self.atom_encoder = nn.Embedding(
            512 * 9 + 1, hidden_dim, padding_idx=0)
        self.edge_encoder = nn.Embedding(
            512 * 3 + 1, num_heads, padding_idx=0)
        self.edge_type = edge_type
        if self.edge_type == 'multi_hop':
            self.edge_dis_encoder = nn.Embedding(
                128 * num_heads * num_heads, 1)
        self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
        self.in_degree_encoder = nn.Embedding(
            512, hidden_dim, padding_idx=0)
        self.out_degree_encoder = nn.Embedding(
            512, hidden_dim, padding_idx=0)

        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.downstream_out_proj = nn.Linear(
            hidden_dim, num_class)

        self.graph_token = nn.Embedding(1, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

        # self.evaluator = get_dataset(dataset_name)['evaluator']
        # self.metric = get_dataset(dataset_name)['metric']
        # self.loss_fn = get_dataset(dataset_name)['loss_fn']
        self.dataset_name = dataset_name

        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.multi_hop_max_dist = multi_hop_max_dist

        self.hidden_dim = hidden_dim
        # self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
        in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
        edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
        # graph_attn_bias
        n_graph, n_node = 1, x.size(0)
        x = x.view(n_graph, n_node, -1)
        attn_bias = attn_bias.view(n_graph,n_node+1,n_node+1)
        graph_attn_bias = attn_bias.clone()
        graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
            1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

        # spatial pos
        # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
        spatial_pos_bias = self.spatial_pos_encoder(
            spatial_pos.view(n_graph, n_node, n_node))
        spatial_pos_bias = spatial_pos_bias.permute(0, 3, 1, 2)
        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                        :, 1:, 1:] + spatial_pos_bias
        # reset spatial pos here
        t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
        graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
        graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

        # edge feature
        if self.edge_type == 'multi_hop':
            spatial_pos_ = spatial_pos.clone()
            spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
            # set 1 to 1, x > 1 to x - 1
            spatial_pos_ = torch.where(
                spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
            if self.multi_hop_max_dist > 0:
                spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
                edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
            # [n_graph, n_node, n_node, max_dist, n_head]
            edge_input = self.edge_encoder(edge_input).mean(-2)
            max_dist = edge_input.size(-2)
            edge_input_flat = edge_input.permute(
                3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
                -1, self.num_heads, self.num_heads)[:max_dist, :, :])
            edge_input = edge_input_flat.reshape(
                max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_input = (edge_input.sum(-2) /
                          (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
        else:
            # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
            edge_input = self.edge_encoder(
                attn_edge_type.view(n_graph, n_node, n_node,-1)).mean(-2).permute(0, 3, 1, 2)

        graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
                                                        :, 1:, 1:] + edge_input
        graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

        # node feauture + graph token
        # node_feature = self.atom_encoder(x)
        # node_feature = node_feature.sum(
        #     dim=-2)           # [n_graph, n_node, n_hidden]

        node_feature = x

        node_feature = node_feature + \
            self.in_degree_encoder(in_degree) + \
            self.out_degree_encoder(out_degree)
        graph_token_feature = self.graph_token.weight.unsqueeze(
            0).repeat(n_graph, 1, 1)
        graph_node_feature = torch.cat(
            [graph_token_feature, node_feature], dim=1)

        # transfomrer encoder
        output = self.input_dropout(graph_node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, graph_attn_bias)
        output = self.final_ln(output)

        # output part

        output = self.downstream_out_proj(output[:, 0, :])
        return output

    def training_step(self, batched_data, batch_idx):

        y_hat = self(batched_data).view(-1)
        y_gt = batched_data.y.view(-1)
        loss = self.loss_fn(y_hat, y_gt)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batched_data, batch_idx):

        y_pred = self(batched_data)
        y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
        }

    def validation_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])

        input_dict = {"y_true": y_true, "y_pred": y_pred}
        try:
            self.log('valid_' + self.metric, self.evaluator.eval(input_dict)
                     [self.metric], sync_dist=True)
        except:
            pass

    def test_step(self, batched_data, batch_idx):

        y_pred = self(batched_data)
        y_true = batched_data.y
        return {
            'y_pred': y_pred,
            'y_true': y_true,
            'idx': batched_data.idx,
        }

    def test_epoch_end(self, outputs):
        y_pred = torch.cat([i['y_pred'] for i in outputs])
        y_true = torch.cat([i['y_true'] for i in outputs])
        input_dict = {"y_true": y_true, "y_pred": y_pred}
        self.log('test_' + self.metric, self.evaluator.eval(input_dict)
                 [self.metric], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
        lr_scheduler = {
            'scheduler': PolynomialDecayLR(
                optimizer,
                warmup_updates=self.warmup_updates,
                tot_updates=self.tot_updates,
                lr=self.peak_lr,
                end_lr=self.end_lr,
                power=1.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Graphormer")
        parser.add_argument('--n_layers', type=int, default=12)
        parser.add_argument('--num_heads', type=int, default=32)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--ffn_dim', type=int, default=512)
        parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
        parser.add_argument('--dropout_rate', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=0.01)
        parser.add_argument('--attention_dropout_rate',
                            type=float, default=0.1)
        parser.add_argument('--checkpoint_path', type=str, default='')
        parser.add_argument('--warmup_updates', type=int, default=60000)
        parser.add_argument('--tot_updates', type=int, default=1000000)
        parser.add_argument('--peak_lr', type=float, default=2e-4)
        parser.add_argument('--end_lr', type=float, default=1e-9)
        parser.add_argument('--edge_type', type=str, default='multi_hop')
        parser.add_argument('--validate', action='store_true', default=False)
        parser.add_argument('--test', action='store_true', default=False)
        return parent_parser


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class CMP(nn.Module):
    def __init__(self, in_channels):
        super(CMP, self).__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            *conv_block(3*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, 2*in_channels, 3, 1, 1, act="leaky"),
            *conv_block(2*in_channels, in_channels, 3, 1, 1, act="leaky"))

    def forward(self, feats, edges=None):
        # allocate memory
        dtype, device = feats.dtype, feats.device
        edges = edges.view(-1, 3)
        V, E = feats.size(0), edges.size(0)
        pooled_v_pos = torch.zeros(
            V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        pooled_v_neg = torch.zeros(
            V, feats.shape[-3], feats.shape[-1], feats.shape[-1], dtype=dtype, device=device)
        # pool positive edges
        pos_inds = torch.where(edges[:, 1] > 0)
        pos_v_src = torch.cat(
            [edges[pos_inds[0], 0], edges[pos_inds[0], 2]]).long()
        pos_v_dst = torch.cat(
            [edges[pos_inds[0], 2], edges[pos_inds[0], 0]]).long()
        pos_vecs_src = feats[pos_v_src.contiguous()]
        pos_v_dst = pos_v_dst.view(-1, 1, 1,
                                   1).expand_as(pos_vecs_src).to(device)
        pooled_v_pos = torch.scatter_add(
            pooled_v_pos, 0, pos_v_dst, pos_vecs_src)
        # pool negative edges
        neg_inds = torch.where(edges[:, 1] < 0)
        neg_v_src = torch.cat(
            [edges[neg_inds[0], 0], edges[neg_inds[0], 2]]).long()
        neg_v_dst = torch.cat(
            [edges[neg_inds[0], 2], edges[neg_inds[0], 0]]).long()
        neg_vecs_src = feats[neg_v_src.contiguous()]
        neg_v_dst = neg_v_dst.view(-1, 1, 1,
                                   1).expand_as(neg_vecs_src).to(device)
        pooled_v_neg = torch.scatter_add(
            pooled_v_neg, 0, neg_v_dst, neg_vecs_src)
        # update nodes features
        enc_in = torch.cat([feats, pooled_v_pos, pooled_v_neg], 1)
        out = self.encoder(enc_in)
        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.init_size = 32 // 4
        self.l1 = nn.Sequential(
            nn.Linear(146, 16 * self.init_size ** 2))  # 146
        self.upsample_1 = nn.Sequential(
            *conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_2 = nn.Sequential(
            *conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.upsample_3 = nn.Sequential(
            *conv_block(16, 16, 4, 2, 1, act="leaky", upsample=True))
        self.cmp_1 = CMP(in_channels=16)
        self.cmp_2 = CMP(in_channels=16)
        self.cmp_3 = CMP(in_channels=16)
        self.cmp_4 = CMP(in_channels=16)
        self.decoder = nn.Sequential(
            *conv_block(16, 256, 3, 1, 1, act="leaky"),
            *conv_block(256, 128, 3, 1, 1, act="leaky"),
            *conv_block(128, 1, 3, 1, 1, act="tanh"))
        # for finetuning
        self.l1_fixed = nn.Sequential(nn.Linear(1, 1 * self.init_size ** 2))
        self.enc_1 = nn.Sequential(
            *conv_block(2, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 32, 3, 2, 1, act="leaky"),
            *conv_block(32, 16, 3, 2, 1, act="leaky"))
        self.enc_2 = nn.Sequential(
            *conv_block(32, 32, 3, 1, 1, act="leaky"),
            *conv_block(32, 16, 3, 1, 1, act="leaky"))

    def forward(self, z, given_m=None, given_y=None, given_w=None, given_v=None):
        # input: (noise, mask, nodes, edges)
        z = z.view(-1, 128)
        # include nodes
        y = given_y.view(-1, 18)
        z = torch.cat([z, y], 1)
        x = self.l1(z)
        f = x.view(-1, 16, self.init_size, self.init_size)
        # combine masks and noise vectors
        m = self.enc_1(given_m)
        f = torch.cat([f, m], 1)
        f = self.enc_2(f)
        # apply Conv-MPN
        x = self.cmp_1(f, given_w).view(-1, *f.shape[1:])
        x = self.upsample_1(x)
        x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
        x = self.upsample_2(x)
        x = self.cmp_3(x, given_w).view(-1, *x.shape[1:])
        x = self.upsample_3(x)
        x = self.cmp_4(x, given_w).view(-1, *x.shape[1:])
        x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
        x = x.view(-1, *x.shape[2:])
        return x


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.encoder = nn.Sequential(
#             *conv_block(9, 16, 3, 1, 1, act="leaky"),
#             *conv_block(16, 16, 3, 1, 1, act="leaky"),
#             *conv_block(16, 16, 3, 1, 1, act="leaky"),
#             *conv_block(16, 16, 3, 1, 1, act="leaky"))
#         self.l1 = nn.Sequential(nn.Linear(18, 8 * 64 ** 2))
#         self.cmp_1 = CMP(in_channels=16)
#         self.downsample_1 = nn.Sequential(
#             *conv_block(16, 16, 3, 2, 1, act="leaky"))
#         self.cmp_2 = CMP(in_channels=16)
#         self.downsample_2 = nn.Sequential(
#             *conv_block(16, 16, 3, 2, 1, act="leaky"))
#         self.cmp_3 = CMP(in_channels=16)
#         self.downsample_3 = nn.Sequential(
#             *conv_block(16, 16, 3, 2, 1, act="leaky"))
#         self.cmp_4 = CMP(in_channels=16)

#         self.decoder = nn.Sequential(
#             *conv_block(16, 256, 3, 2, 1, act="leaky"),
#             *conv_block(256, 128, 3, 2, 1, act="leaky"),
#             *conv_block(128, 128, 3, 2, 1, act="leaky"))
#         # The height and width of downsampled image
#         ds_size = 32 // 2 ** 4
#         self.fc_layer_global = nn.Sequential(nn.Linear(128, 1))
#         self.fc_layer_local = nn.Sequential(nn.Linear(128, 1))

#     def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
#           #mks, given_nds, given_eds, nd_to_sample
#         x = x.view(-1, 1, 64, 64)

#         # include nodes
#         y = given_y
#         y = self.l1(y)
#         y = y.view(-1, 8, 64, 64)
#         x = torch.cat([x, y], 1)
#         # message passing -- Conv-MPN
#         x = self.encoder(x)
#         x = self.cmp_1(x, given_w).view(-1, *x.shape[1:])
#         x = self.downsample_1(x)
#         x = self.cmp_2(x, given_w).view(-1, *x.shape[1:])
#         x = self.downsample_2(x)
#         x = self.cmp_3(x, given_w).view(-1, *x.shape[1:])
#         x = self.downsample_3(x)
#         x = self.cmp_4(x, given_w).view(-1, *x.shape[1:])
#         x = self.decoder(x.view(-1, x.shape[1], *x.shape[2:]))
#         x = x.view(-1, x.shape[1])
#         # global loss
#         x_g = add_pool(x, nd_to_sample)
#         validity_global = self.fc_layer_global(x_g)
#         return validity_global


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.encoder = Graphormer(n_layers=12,
                                  num_heads=32,
                                  hidden_dim=512,
                                  dropout_rate=0.1,
                                  intput_dropout_rate=0.1,
                                  weight_decay=0.01,
                                  ffn_dim=512,
                                  dataset_name="houseplan",
                                  warmup_updates=60000,
                                  tot_updates=1000000,
                                  peak_lr=2e-4,
                                  end_lr=1e-9,
                                  edge_type='null',
                                  multi_hop_max_dist=20,
                                  attention_dropout_rate=0.1,
                                  num_class=1)
        self.l1 = nn.Sequential(nn.Linear(18,  64 ** 2))
        self.l2 = nn.Sequential(nn.Linear(2*64**2,  512))

    def forward(self, x, given_y=None, given_w=None, nd_to_sample=None):
        #mks, given_nds, given_eds, nd_to_sample
        x = x.view(-1, 64 * 64)

        # include nodes
        y = given_y
        y = self.l1(y)
        y = y.view(-1, 64 * 64)
        x = torch.cat([x, y], 1).to(device)
        x = self.l2(x)

        # prepare edge_attr and edge_index
        indice = (given_w[:, 1] == 1).nonzero()
        # COO format, [2,num_edges]
        edge_index = given_w[:, (0, 2)][indice].view(-1, 2).T.to(device)
        edge_attr = torch.ones(
            edge_index.shape[1], dtype=torch.long).view(-1, 1).to(device)
        item = OutputItem(edge_index, edge_attr, x)
        item = preprocess_item(item)
        return self.encoder(item)


class OutputItem():
    def __init__(self, edge_index, edge_attr, x):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.x = x
