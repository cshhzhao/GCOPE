import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from fastargs.decorators import param

class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group , group_num=1, inner_prune=None, edge_attr_dim=0): #这里的prompt graph与dataset中的Graph没有边的连接，所以这里group num就是prompt Graph的数量，每一个PG都有token_num个tokens
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.edge_attr_dim = edge_attr_dim
        self.inner_prune = inner_prune #用于更新prompt Graph的内部结构，假入一组tokens中两个tokens之间的similarity＜inner_prune,那么他们两个之间的边就要删除。

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)]) #如果要求输入数据是可训练的，就通过torch.nn.Parameter类对象的实例化定义
        
        if(edge_attr_dim!=0):
            self.edge_token_list = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.empty(1, edge_attr_dim)) for i in range(1)]) #如果要求输入数据是可训练的，就通过torch.nn.Parameter类对象的实例化定义

        self.token_init(init_method="kaiming_uniform") #使用kaiming_uniform的方法进行参数的初始化

    def token_init(self, init_method="kaiming_uniform"):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
            
            if(self.edge_attr_dim !=0 ):
                for edge_token in self.edge_token_list:
                    torch.nn.init.kaiming_uniform_(edge_token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")

    def inner_structure_update(self):
        if(self.edge_attr_dim!=0):
            return self.token_view_with_edge_attr()
        else:
            return self.token_view_without_edge_attr()
    
    # 带边属性的版本
    def token_view_with_edge_attr(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list): #注意哈，token_list的存在是因为，prompt graphs的数量可以不为1；但是如果是要建立prompt graph和input graphs的连接，则len(token_list)=1，也就是prompt graph只有一个
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1)) #矩阵相乘
            token_sim = torch.sigmoid(token_dot)  # 0-1 tokens之间的相似性（tokens就是一组token节点）

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim) #如果当前prompt Graph中两个token节点之间有相似性，那么就增加边。
            edge_index = inner_adj.nonzero().t().contiguous()
            edge_attr = None # 是一个tensor对象
            for current_index, current_edge_start in enumerate(edge_index[0]):
                token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
                if(len(tokens[0].shape)==1):
                    current_edge_attr = self.edge_token_list[0]*torch.mm(tokens[current_edge_start].reshape((1,1)),tokens[edge_index[1][current_index]].reshape((1,1)))
                else:
                    current_edge_attr = self.edge_token_list[0]*torch.mm(tokens[current_edge_start],tokens[edge_index[1][current_index]])

                if(current_index == 0):
                    edge_attr=current_edge_attr
                else:
                    edge_attr = torch.cat((edge_attr,current_edge_attr),dim=0)
            
            edge_attr = F.leaky_relu(edge_attr) #限制一下边属性的范围，不然可tune空间太大了？？？

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long(), edge_attr=edge_attr))

        # 返回后需要再heavyprompt类下面更新positional encoding
        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch

    def token_view_without_edge_attr(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list): #注意哈，token_list的存在是因为，prompt graphs的数量可以不为1；但是如果是要建立prompt graph和input graphs的连接，则len(token_list)=1，也就是prompt graph只有一个
            # inner link: token-->token
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1)) #矩阵相乘
            token_sim = torch.sigmoid(token_dot)  # 0-1 tokens之间的相似性（tokens就是一组token节点）

            inner_adj = torch.where(token_sim < self.inner_prune, 0, token_sim) #如果当前prompt Graph中两个token节点之间有相似性，那么就增加边。
            edge_index = inner_adj.nonzero().t().contiguous()            

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))

        # 返回后需要再heavyprompt类下面更新positional encoding
        pg_batch = Batch.from_data_list(pg_list)
        return pg_batch    


class HeavyPrompt(LightPrompt): #一个HeavyPrompt对应一个target dataset
    def __init__(self, token_dim, token_num, cross_prune=0.1, inner_prune=0.01, edge_attr_dim=0):
        super(HeavyPrompt, self).__init__(token_dim, token_num, 1, inner_prune, edge_attr_dim)  # only has one prompt graph.
        self.cross_prune = cross_prune # cross_prune是剪边的阈值，加入两个node之间feature vector的相似性小于(＜)cross_prune的值，那么边就删减
        self.edge_attr_dim = edge_attr_dim

    def forward(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        if(self.edge_attr_dim == 0):

            # 之所以要更新prompt graphs，是因为在上一个batch下，经过loss.backward()后tokens的参数更新了，但是PG的结构并没有更新。
            pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

            # trick：对于所有input graph来说，prompt graph的内部结构是一样的，区别是prompt Graph和input graph的外部连接的区别。

            inner_edge_index = pg.edge_index
            token_num = pg.x.shape[0]

            re_graph_list = []
            for g in Batch.to_data_list(graph_batch):
                g_edge_index = g.edge_index + token_num # 将token的节点编码到前面，也就是原图的0-3节点，引入2个prompt token node(0-1编码)后编码是2-5
                
                cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
                cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
                cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
                
                cross_edge_index = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息            
                cross_edge_index[1] = cross_edge_index[1] + token_num
                
                x = torch.cat([pg.x, g.x], dim=0)
                y = g.y

                edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
                data = Data(x=x, edge_index=edge_index, y=y)
                re_graph_list.append(data)
            
            graph_batch = Batch.from_data_list(re_graph_list)

            return graph_batch    
        else:

            # 之所以要更新prompt graphs，是因为在上一个batch下，经过loss.backward()后tokens的参数更新了，但是PG的结构并没有更新。
            pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

            # trick：对于所有input graph来说，prompt graph的内部结构是一样的，区别是prompt Graph和input graph的外部连接的区别。

            inner_edge_index = pg.edge_index
            token_num = pg.x.shape[0]

            re_graph_list = []
            for g in Batch.to_data_list(graph_batch):
                g_edge_index = g.edge_index + token_num # 将token的节点编码到前面，也就是原图的0-3节点，引入2个prompt token node(0-1编码)后编码是2-5
                
                cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
                cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
                cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
                
                cross_edge_index = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息
                
                cross_edge_attr = None # 是一个tensor对象
                for current_index, current_edge_start in enumerate(cross_edge_index[0]):
                    
                    # 如果节点的特征只有1维，那么取对应的节点特征是tensor(a)，不是matrix类型，不能用下面的torch.mm()函数，所以要改一下
                    if(pg.x.shape[1]==1):
                        current_edge_attr = self.edge_token_list[0]*torch.mm(pg.x[current_edge_start].reshape((1,1)),g.x[cross_edge_index[1][current_index]].reshape((1,1)))
                    else:
                        current_edge_attr = self.edge_token_list[0]*torch.mm(pg.x[current_edge_start],g.x[cross_edge_index[1][current_index]])

                    if(current_index == 0):
                        cross_edge_attr=current_edge_attr
                    else:
                        cross_edge_attr = torch.cat((cross_edge_attr,current_edge_attr),dim=0)            
                
                cross_edge_attr = F.leaky_relu(cross_edge_attr)
                
                cross_edge_index[1] = cross_edge_index[1] + token_num
                
                x = torch.cat([pg.x, g.x], dim=0)
                y = g.y

                edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
                edge_attr = torch.cat([pg.edge_attr, g.edge_attr, cross_edge_attr], dim=0)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                re_graph_list.append(data)

            # 对下面这段代码需要重新计算一个pe表示
            graph_batch = Batch.from_data_list(re_graph_list)

            return graph_batch            
    
    def modify_origianl_data(self, graph_list):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        # device = torch.device("cuda")
        # device = torch.device("cpu")

        # 之所以要更新prompt graphs，是因为在上一个batch下，经过loss.backward()后tokens的参数更新了，但是PG的结构并没有更新。
        pg = self.inner_structure_update()  # batch of prompt graph (currently only 1 prompt graph in the batch)

        # trick：对于所有input graph来说，prompt graph的内部结构是一样的，区别是prompt Graph和input graph的外部连接的区别。

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in graph_list:
            g_edge_index = g.edge_index + token_num
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < self.cross_prune, 0, cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        return re_graph_list


@param('adapt.prompt.cross_prune')
@param('adapt.prompt.inner_prune')
@param('adapt.prompt.edge_attr_dim')
def get_prompt_model(num_features, prompt_node_num, cross_prune, inner_prune, edge_attr_dim):
    return HeavyPrompt(num_features, prompt_node_num, cross_prune, inner_prune, edge_attr_dim) # HeavyPrompt是带结构的，对应了一个prompt graph。