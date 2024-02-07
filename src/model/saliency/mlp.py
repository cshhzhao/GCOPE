import torch


class MLPSaliency(torch.nn.Module):
    def __init__(self, feature_dim, hid_dim, layer_num):
        super().__init__()
        
        self.lins = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(layer_num):
            if i == 0:
                self.bns.append(torch.nn.BatchNorm1d(feature_dim))
                self.lins.append(torch.nn.Linear(feature_dim, hid_dim))
            elif i == layer_num - 1:
                self.bns.append(torch.nn.BatchNorm1d(hid_dim))
                self.lins.append(torch.nn.Linear(hid_dim, feature_dim))
            else:
                self.bns.append(torch.nn.BatchNorm1d(hid_dim))
                self.lins.append(torch.nn.Linear(hid_dim, hid_dim))
        
        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)        
    
    def forward(self, x):
        
        x_ori = x.clone()

        for i, (lin, bn) in enumerate(zip(self.lins, self.bns)):
            x = bn(x)
            x = torch.relu(lin(x)) if i != len(self.lins) - 1 else lin(x)
            
        return torch.sigmoid(x) * x_ori
    

from fastargs.decorators import param

@param('model.saliency.mlp.hid_dim')
@param('model.saliency.mlp.num_layers')
def get_model(feature_dim, hid_dim, num_layers):
    return MLPSaliency(feature_dim, hid_dim, num_layers)