import torch
from functools import partial
from torch_geometric.nn import global_add_pool
from fastargs.decorators import param
from model.backbone.gcn_conv import GCNConv
from torch.nn import BatchNorm1d
import torch.nn.functional as F
import pdb

class GCN(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_features,
                       hidden, num_conv_layers=3,
                       num_feat_layers=1, gfn=False, collapse=False, residual=False,
                       res_branch="BNConvReLU", dropout=0, 
                       edge_norm=True):
        super(GCN, self).__init__()

        self.global_pool = global_add_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_features
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)
    
    @param('general.reconstruct')
    def forward(self, data, reconstruct):
        
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        h = self.bn_feat(x)
        h = F.relu(self.conv_feat(h, edge_index))
        
        for i, conv in enumerate(self.convs):
            h = self.bns_conv[i](h)
            h = F.relu(conv(h, edge_index))
            
        graph_emb = self.global_pool(h, batch)
        
        if(reconstruct==0.0):
            return graph_emb
        else:
            return graph_emb, h

@param('model.backbone.hid_dim')
@param('model.backbone.gcn.num_conv_layers')
@param('model.backbone.gcn.dropout')
def get_model(num_features, hid_dim, num_conv_layers, dropout):
    return GCN(num_features=num_features, hidden=hid_dim, num_conv_layers=num_conv_layers, dropout=dropout, gfn=False)

# 实例化介绍
# self.GraphModel = GCN(num_features=input_dim, hidden=hid_dim,
#                         num_conv_layers=gnn_layer_num,
#                         dropout=dropout, gfn=False)