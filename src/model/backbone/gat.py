import torch
from torch_geometric.nn import global_add_pool, GATConv
from fastargs.decorators import param
from model.backbone.gcn_conv import GCNConv
from torch.nn import BatchNorm1d
import torch.nn.functional as F

class GAT(torch.nn.Module):
    def __init__(self, num_features, 
                       hidden,
                       head=4,
                       num_conv_layers=3, 
                       dropout=0.2):

        super(GAT, self).__init__()
        self.hidden_dim = hidden
        self.global_pool = global_add_pool
        self.dropout = dropout
        hidden_in = num_features
   
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GATConv(hidden, int(hidden / head), heads=head, dropout=dropout))

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
        # h = F.relu(self.conv_feat(h, edge_index))
        h = F.leaky_relu(self.conv_feat(h, edge_index))
        
        for i, conv in enumerate(self.convs):
            h = self.bns_conv[i](h)
            # h = F.relu(conv(h, edge_index))
            h = F.leaky_relu(conv(h, edge_index))

        graph_emb = self.global_pool(h, batch)

        if(reconstruct==0.0):
            return graph_emb
        else:
            return graph_emb, h

@param('model.backbone.hid_dim')
@param('model.backbone.gat.head')
@param('model.backbone.gat.num_conv_layers')
@param('model.backbone.gat.dropout')
def get_model(num_features, hid_dim, head, num_conv_layers, dropout):
    return GAT(num_features=num_features, hidden=hid_dim, head=head, num_conv_layers=num_conv_layers, dropout=dropout)

# 实例化介绍
# self.head = head #只有GAT模型的时候才保存head参数
# self.GraphModel = GAT(num_features=input_dim, hidden=hid_dim,
#                         head=head, num_conv_layers=gnn_layer_num,
#                         dropout=dropout)    