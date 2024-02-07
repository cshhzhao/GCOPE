import torch


class MLPAnswering(torch.nn.Module):
    def __init__(self, hid_dim, num_class, answering_layer_num):
        super().__init__()
        self.answering_layer_num = answering_layer_num
        self.num_class = num_class
        
        self.answering = torch.nn.ModuleList()
        self.bns_answer = torch.nn.ModuleList()

        for i in range(answering_layer_num-1):
            self.bns_answer.append(torch.nn.BatchNorm1d(hid_dim))
            self.answering.append(torch.nn.Linear(hid_dim,hid_dim))
        
        self.bn_hid_answer = torch.nn.BatchNorm1d(hid_dim)
        self.final_answer = torch.nn.Linear(hid_dim, num_class)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)        
    
    def forward(self, x):
        
        for i, lin in enumerate(self.answering):
            x = self.bns_answer[i](x)
            x = torch.relu(lin(x))
            
        x = self.bn_hid_answer(x)
        x = self.final_answer(x)
        prediction = torch.log_softmax(x, dim=-1)
        return prediction
    

from fastargs.decorators import param

@param('model.backbone.hid_dim')
@param('model.answering.mlp.num_layers')
def get_model(hid_dim, num_class, num_layers):
    return MLPAnswering(hid_dim, num_class, num_layers)