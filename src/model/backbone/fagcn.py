import torch
from torch_geometric.nn import global_add_pool, FAConv
from fastargs.decorators import param

class FAGCN(torch.nn.Module):

    def __init__(self, num_features, hidden, num_conv_layers, dropout, epsilon):
        super(FAGCN, self).__init__()
        self.global_pool = global_add_pool
        self.eps = epsilon
        self.layer_num = num_conv_layers
        self.dropout = dropout
        self.hidden_dim = hidden

        self.layers = torch.nn.ModuleList()
        for _ in range(self.layer_num):
            self.layers.append(FAConv(hidden, epsilon, dropout))

        self.t1 = torch.nn.Linear(num_features, hidden)
        self.t2 = torch.nn.Linear(hidden, hidden)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        torch.nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    @param('general.reconstruct')
    def forward(self, data, reconstruct):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch

        h = torch.dropout(x, p=self.dropout, train=self.training)
        h = torch.relu(self.t1(h))
        h = torch.dropout(h, p=self.dropout, train=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, raw, edge_index)
        h = self.t2(h)
        graph_emb = self.global_pool(h, batch)

        if(reconstruct==0.0):
            return graph_emb
        else:
            return graph_emb, h


from fastargs.decorators import param

@param('model.backbone.hid_dim')
@param('model.backbone.fagcn.num_conv_layers')
@param('model.backbone.fagcn.dropout')
@param('model.backbone.fagcn.epsilon')
def get_model(num_features, hid_dim, num_conv_layers, dropout, epsilon):
    return FAGCN(num_features, hid_dim, num_conv_layers, dropout, epsilon)