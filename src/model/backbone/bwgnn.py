import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import math
import dgl
import sympy
import scipy
import numpy as np
from torch import nn
from torch.nn import init
import torch_geometric.utils as pyg_utils
import networkx as nx
from fastargs.decorators import param

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


class PolyConvBatch(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConvBatch, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, block, feat):
        def unnLaplacian(feat, D_invsqrt, block):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            block.srcdata['h'] = feat * D_invsqrt
            block.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - block.srcdata.pop('h') * D_invsqrt

        with block.local_scope():
            D_invsqrt = torch.pow(block.out_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, block)
                h += self._theta[k]*feat
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.thetas = calculate_theta2(d=d)
        self.hidden_dim = hidden_dim
        self.conv = []
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(hidden_dim, hidden_dim, self.thetas[i], lin=False))
            else:
                self.conv.append(PolyConvBatch(hidden_dim, hidden_dim, self.thetas[i], lin=False))
        self.linear = nn.Linear(num_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim*len(self.conv), hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        self.d = d
    
    @param('general.reconstruct')
    def forward(self, batch_pyg_graphs, reconstruct):
        # 1、将batch_pyg_g遍历，每一步都转换为dgl图
        # 2、将每一次得到的节点表示
        device = next(self.parameters()).device
        networkx_graphs = [pyg_utils.to_networkx(pyg_graph, node_attrs=['x']) for pyg_graph in batch_pyg_graphs.to_data_list()]
        dgl_graphs = [dgl.from_networkx(networkx_graph, node_attrs=['x']) for networkx_graph in networkx_graphs]
        batch_dgl_graph = dgl.batch(dgl_graphs).to(device)
        in_feat = batch_dgl_graph.ndata['x']
        h_rep = self.linear(in_feat)
        h_rep = self.act(h_rep)
        h_rep = self.linear2(h_rep)
        h_rep = self.act(h_rep)
        h_final = torch.zeros([len(in_feat), 0]).to(device)
        for conv in self.conv:
            h0 = conv(batch_dgl_graph, h_rep)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h_rep = self.linear3(h_final)
        h_rep = self.act(h_rep)
        h_rep = self.linear4(h_rep)
        batch_dgl_graph.ndata['x'] = h_rep
        dgl_graph_list = dgl.unbatch(batch_dgl_graph)
        graph_embs = torch.cat([dgl.sum_nodes(g, 'x') for g in dgl_graph_list],dim=0)
        if(reconstruct==0.0):
            return graph_embs
        else:
            return graph_embs, h_rep

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(in_feat),0])
        for conv in self.conv:
            h0 = conv(blocks[0], h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

@param('model.backbone.hid_dim')
def get_model(num_features, hid_dim):
    return BWGNN(num_features, hid_dim)