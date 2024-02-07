import torch
import torch.nn as nn
from copy import deepcopy

class VirtualNode(nn.Module):
    def __init__(self, num_node_features, num_virtual_nodes):
        super(VirtualNode, self).__init__()
        # 定义一个可学习的参数
        # self.learnable_param = nn.Parameter(torch.randn(num_node_features))
        self.learnable_param = nn.ParameterList([nn.Parameter(torch.randn(num_node_features)) for _ in range(num_virtual_nodes)])
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]

    # with torch.no_grad()函数，会导致参数不更新
    # def add_learnable_features(self, original_node_features):
    #     # 举例：将所有参数求和，然后与输入 x 相乘 
    #     # virtualNode model向前传播的时候
    #     virtual_node_features = torch.cat([p.reshape((1,-1)) for p in self.learnable_param], dim=0)
    #     updated_features = torch.cat([original_node_features, virtual_node_features], dim=0)
    #     return updated_features
    
    def add_learnable_features_with_no_grad(self, original_node_features):
        # 举例：将所有参数求和，然后与输入 x 相乘 
        # virtualNode model向前传播的时候
        virtual_node_features = torch.cat([p.data.reshape((1,-1)) for p in self.learnable_param], dim=0)
        updated_features = torch.cat([original_node_features, virtual_node_features], dim=0)
        return updated_features

    # 定义了一个不带梯度的序列，保存的是上次更新时的节点参数，也就起到了每个batch的图的virtual nodes更新的都是最新参数，避免了第一个batch更新参数后，其他batch无法匹配最新的learnable params问题
    def forward(self, batch_with_no_grad_node_features):
        count = 0
        graph_index_list = [x for x in set(batch_with_no_grad_node_features.batch.tolist())]
        for graph_index in graph_index_list:
            node_indices_corresponding_graph = (batch_with_no_grad_node_features.batch == graph_index).nonzero(as_tuple=False).view(-1)
            for node_indice in reversed(node_indices_corresponding_graph):
                for index, param_value in enumerate(self.last_updated_param):
                    if(torch.equal(batch_with_no_grad_node_features.x[node_indice], param_value)):
                        batch_with_no_grad_node_features.x[node_indice] = self.learnable_param[index]
                        count+=1
        batch_with_learnable_param = batch_with_no_grad_node_features
        # print(f'更新了{count}个learnable param')
        # import pdb
        # pdb.set_trace()

        return batch_with_learnable_param

    def update_last_params(self):
        self.last_updated_param = [deepcopy(param.data) for param in self.learnable_param]

    # alternate_no_grad_features_with_learnable_params 向前传播函数的目标是将图数据中的一些非梯度的向量替换为可学习参数
    # def forward(self, batch_with_no_grad_node_features):
        
    #     count = 0        

    #     graph_index_list = [x for x in set(batch_with_no_grad_node_features.batch.tolist())]
    #     for graph_index in graph_index_list:
    #         maximum_match_param_num = len(self.learnable_param)
    #         node_indices_corresponding_graph = (batch_with_no_grad_node_features.batch == graph_index).nonzero(as_tuple=False).view(-1)
    #         for node_indice in reversed(node_indices_corresponding_graph):
    #             if(count == maximum_match_param_num):
    #                 break
    #             for param in self.learnable_param:
    #                 if(torch.equal(batch_with_no_grad_node_features.x[node_indice], param.data)):                
    #                     batch_with_no_grad_node_features.x[node_indice] = param
    #                     count+=1
    #     batch_with_learnable_param = batch_with_no_grad_node_features

    #     return batch_with_learnable_param

# 示例用法
if __name__ == '__main__':
    model = VirtualNode(num_node_features=10, num_virtual_nodes=5)
    input_tensor = torch.randn(10)

    output = model(input_tensor)
    print(output)