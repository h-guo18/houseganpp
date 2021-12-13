import torch
import numpy as np

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
import algos
import pickle
import joblib
if torch.cuda.is_available():
    device = torch.device('cuda:0')


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long).to(device)
    x = x + feature_offset
    return x


def preprocess_item(item):
    edge_attr, edge_index, x = item.edge_attr.to(device), item.edge_index.to(device), item.x.to(device)
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool).to(device)
    adj[edge_index[0, :], edge_index[1, :]] = True
    #edge_index:(2 x _)

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long).to(device)
    attn_edge_type[edge_index[0, :], edge_index[1, :]
                   ] = convert_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.cpu().numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.cpu().numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long().to(device)
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float).to(device)  # with graph token

    # combine
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos  #matrix recording shortest path distance
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = adj.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long().to(device)

    return item
