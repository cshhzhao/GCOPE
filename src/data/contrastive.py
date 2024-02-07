import time
from fastargs.decorators import param
import torch
from torch_geometric.data import Batch
import numpy as np

# 单向边无法进行随机游走！！！
@param('general.cache_dir')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('pretrain.cross_link_ablation')
@param('pretrain.dynamic_edge')
@param('pretrain.dynamic_prune')
@param('pretrain.split_method')
def get_clustered_data(dataset, cache_dir, num_parts, cross_link, cl_init_method='learnable', cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):

    from .utils import preprocess, iterate_datasets
    data_list = [preprocess(data) for data in iterate_datasets(dataset)]
    from torch_geometric.data import Batch
    data = Batch.from_data_list(data_list)
    from copy import deepcopy

    data_for_similarity_computation = deepcopy(data)

    print(f'Isolated graphs中共有节点{data.num_nodes}个，每个图需要添加{cross_link}个虚点')
    vr_model = None
    # 确定是否要在不同图之间创建连接  
    if(cross_link > 0):
        num_graphs = data.num_graphs
        graph_node_indices = []

        for graph_index in range(num_graphs):
            # 找出属于当前图的节点
            node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_node_indices.append(node_indices)
        # 获取所有Batch graph的点的索引，graph_node_indices列表长度就是Batch的图的数量

        new_index_list = [i for i in range(num_graphs)]*cross_link #cross_link控制了每个图是否有virtual nodes(cross_link=0 or >0),如果有虚点，有几个。
        
        if(cl_init_method == 'mean'):
            new_node_features = []
            # 初始化节点特征，new_index_list是batch graph对象中的图的索引
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                new_node_features.append(torch.mean(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

            new_node_features = torch.cat(new_node_features, dim=0)
            # 将新加入的节点信息融入源图中
            data.x = torch.cat([data.x, new_node_features], dim=0)

        elif(cl_init_method == 'sum'):
            new_node_features = []
            # 初始化节点特征，new_index_list是batch graph对象中的图的索引
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                new_node_features.append(torch.sum(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

            new_node_features = torch.cat([new_node_features], dim=0)
            # 将新加入的节点信息融入源图中
            data.x = torch.cat([data.x, new_node_features], dim=0)
        elif(cl_init_method == 'simple'):
            # 初始化节点特征，为了避免梯度为0，节点特征全部设置为1 
            new_node_features = torch.ones((len(new_index_list),data.num_node_features))
            # 将新加入的节点信息融入源图中
            data.x = torch.cat([data.x, new_node_features], dim=0) 
        elif(cl_init_method == 'learnable'):
            from model.virtual_node_model import VirtualNode               
            vr_model = VirtualNode(data.num_node_features,len(new_index_list))
            # 将新加入的节点信息融入源图中
            data.x = vr_model.add_learnable_features_with_no_grad(data.x)
      
        # 将节点添加到batch对象中，指定新加入的节点，分别属于哪一张图。
        data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)

        if(dynamic_edge=='none'):
            # 每个virtual node和对应的图，比如cora、wisconsin的节点全连接
            if(cross_link==1):            
                for node_graph_index in new_index_list:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index = node_indices_corresponding_graph[-1]
                    
                    # 添加单向边
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                    
                    # 添加双向边
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        # 添加单向边
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        # 添加双向边
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)                                    
            
            # 如果不执行Ablation Study，virtual nodes之间需要连接，进而将所有的预训练图数据都连通成一个大图
            if(cross_link_ablation==False):
                # virtual nodes之间全连接。注意是双向的即可。
                all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
                for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
                    other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index] #双向连接         
                    # other_added_node_index_list = [index for index in all_added_node_index[list_index+1:]] # 单向链接，和pyg的dataset保持一致
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                # 把冗余的边删掉，现在是1->2, 2->1，删掉其中一个。
        elif(dynamic_edge=='internal_external'):
            all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
            cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

            all_cross_edge = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息
           
            vr_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index)) #点为虚点则为True，否则为False
            vr_edge_indices = torch.where(vr_edge_bool_signs)[0] # 找出值为True的边索引值
            vr_cross_edge=all_cross_edge[:,vr_edge_indices] #筛选出存在virtual nodes的边
            vr_cross_undirected_edge = torch.sort(vr_cross_edge, dim=0)[0] #根据入点比出点小的原则对shape=(2,xxx)边对象进行排序，转换成无向图边序列
            vr_cross_undirected_edge_np = vr_cross_undirected_edge.numpy() # 将vr_cross_undirected_edge转换为numpy对象
            vr_cross_unique_edges = np.unique(vr_cross_undirected_edge_np, axis=1) # 去除序列中冗余的边，保证1 -> 2,而不出现2 -> 1

            print(f"Added Edge Num is {len(vr_cross_unique_edges[0])}")

            data.edge_index = torch.cat([data.edge_index, torch.tensor(vr_cross_unique_edges).contiguous()], dim=1)   
        elif(dynamic_edge=='similarity'):

            # 每个virtual node和对应的图，比如cora、wisconsin的节点全连接
            if(cross_link==1):            
                for node_graph_index in new_index_list:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index = node_indices_corresponding_graph[-1]
                    
                    # 添加单向边
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                    
                    # 添加双向边
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        # 添加单向边
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        # 添加双向边
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)     
                                    
            graph_mean_features = []
            # 初始化节点特征，new_index_list是batch graph对象中的图的索引
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph_for_simi = (data_for_similarity_computation.batch == node_graph_index).nonzero(as_tuple=False).view(-1)              
                graph_mean_features.append(torch.mean(data_for_similarity_computation.x[node_indices_corresponding_graph_for_simi],dim=0).tolist())

            graph_mean_features = torch.tensor(graph_mean_features)
            cross_dot = torch.mm(graph_mean_features, torch.transpose(graph_mean_features, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
            cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

            all_cross_edge = cross_adj.nonzero().t().contiguous()
            all_cross_edge += data_for_similarity_computation.num_nodes
            
            total_edge_num_before_add_cross_link_internal_edges = data.edge_index.shape[1]
            # print(f'虚点间连接前的边是{data.edge_index.shape[1]}个')
            # print(f'应该添加{all_cross_edge.shape[1]}条边')
            data.edge_index = torch.cat([data.edge_index, all_cross_edge], dim=1)               
            if((data.edge_index.shape[1]-total_edge_num_before_add_cross_link_internal_edges)==all_cross_edge.shape[1]):
                print(f'虚点间连接后的边是{data.edge_index.shape[1]}个,共增加了{all_cross_edge.shape[1]}条虚点间的连边')

    print(f'预处理后的Graphs中共有节点{data.num_nodes}个，每个图包含{cross_link}个虚点')

    raw_data = deepcopy(data) #如果加虚点了，就是带虚点的图，否则就是原图

    if(split_method == 'metis'):
        import os
        from torch_geometric.loader.cluster import ClusterData

        metis_cache_dir = os.path.join(cache_dir, ','.join(dataset))
        os.makedirs(metis_cache_dir, exist_ok=True)

        data = list(ClusterData(data, num_parts=num_parts, save_dir=metis_cache_dir))

        return data, vr_model, raw_data
    elif(split_method=='RandomWalk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = []  # 存储子图的列表
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

        return graph_list, vr_model, raw_data

@param('pretrain.cross_link_ablation')
@param('pretrain.dynamic_edge')
@param('pretrain.dynamic_prune')
@param('pretrain.split_method')
# 每次都需要重新切分，因为参数更新后，没有办法再和原来的节点进行匹配了，除非提前记录下哪个id和那个参数匹配
def get_clustered_data_for_learnable_vr(data, num_parts, vr_model, cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):

    # 确定是否要在不同图之间创建连接
    num_graphs = data.num_graphs
    graph_node_indices = []
    if(dynamic_edge=='none'):
        vr_node_index_list = list(range(data.num_nodes - num_graphs, data.num_nodes))
        
        for index, vr_node_index in enumerate(vr_node_index_list):
            data.x[vr_node_index] = vr_model.learnable_param[index].data.reshape((1,-1))

    for graph_index in range(num_graphs):
        # 找出属于当前图的节点
        node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
        graph_node_indices.append(node_indices)
    # 获取所有Batch graph的点的索引，graph_node_indices列表长度就是Batch的图的数量

    new_index_list = [i for i in range(data.num_graphs)]
    
    # 将新加入的节点信息融入源图中
    data.x = vr_model.add_learnable_features_with_no_grad(data.x)
    # data.x = vr_model.add_learnable_features(data.x)
    
    # 将节点添加到batch对象中，指定新加入的节点，分别属于哪一张图。
    data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)
    if(dynamic_edge=='none'):
        # 每个virtual node和对应的图，比如cora、wisconsin的节点全连接
        for node_graph_index in new_index_list:
            node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
            new_node_index = node_indices_corresponding_graph[-1]
            
            # 添加单向边
            new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
            data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            
            # 添加双向边
            new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
            data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
        
        # 如果不执行Ablation Study，virtual nodes之间需要连接，进而将所有的预训练图数据都连通成一个大图
        if(cross_link_ablation==False):
            # virtual nodes之间全连接。注意是双向的即可。
            all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
            for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
                other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index] #双向连接         
                # other_added_node_index_list = [index for index in all_added_node_index[list_index+1:]] # 单向链接，和pyg的dataset保持一致
                new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
                data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            # 把冗余的边删掉，现在是1->2, 2->1，删掉其中一个。

    elif(dynamic_edge=='internal_external'):            
        all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
        cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
        cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
        cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

        all_cross_edge = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息
        
        vr_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index)) #点为虚点则为True，否则为False
        vr_edge_indices = torch.where(vr_edge_bool_signs)[0] # 找出值为True的边索引值
        vr_cross_edge=all_cross_edge[:,vr_edge_indices] #筛选出存在virtual nodes的边
        vr_cross_undirected_edge = torch.sort(vr_cross_edge, dim=0)[0] #根据入点比出点小的原则对shape=(2,xxx)边对象进行排序，转换成无向图边序列
        vr_cross_undirected_edge_np = vr_cross_undirected_edge.numpy() # 将vr_cross_undirected_edge转换为numpy对象
        vr_cross_unique_edges = np.unique(vr_cross_undirected_edge_np, axis=1) # 去除序列中冗余的边，保证1 -> 2,而不出现2 -> 1

        print(f"Added Edge Num is {len(vr_cross_unique_edges[0])}")

        data.edge_index = torch.cat([data.edge_index, torch.tensor(vr_cross_unique_edges).contiguous()], dim=1)

    if(split_method == 'metis'):
        import os
        from torch_geometric.loader.cluster import ClusterData

        data = list(ClusterData(data, num_parts=num_parts))
        return data
    
    elif(split_method=='RandomWalk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = []  # 存储子图的列表
        skip_num = 0
        for walk in walk_list:
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        return graph_list

# 更新当前graph list中的虚点参数
def update_graph_list_param(graph_list, vr_model):
    
    # v1   已验证  跑完的实验代码 ！！！！只是执行慢一点，但是最精准
    # count = 0
    # batch_with_no_grad_node_features = Batch.from_data_list(graph_list)
    # graph_index_list = [x for x in set(batch_with_no_grad_node_features.batch.tolist())]
    # for graph_index in graph_index_list:
    #     node_indices_corresponding_graph = (batch_with_no_grad_node_features.batch == graph_index).nonzero(as_tuple=False).view(-1)
    #     for node_indice in reversed(node_indices_corresponding_graph):
    #         for index, param_value in enumerate(vr_model.last_updated_param):
    #             if(torch.equal(batch_with_no_grad_node_features.x[node_indice], param_value)):
    #                 batch_with_no_grad_node_features.x[node_indice] = vr_model.learnable_param[index].data
    #                 # print(f'匹配成功，第{index}个learnable params')
    #                 count+=1
    # batch_with_newest_learnable_param = batch_with_no_grad_node_features
    # updated_graph_list = batch_with_newest_learnable_param.to_data_list()
    # return updated_graph_list

    # 上下两个版本一样，下面的这段代码，可以一定程度上加速执行速度 v2  已验证
    # count = 0
    # start_time = time.time()
    # for graph_index, graph in enumerate(graph_list):
    #     for node_indice in range(graph.num_nodes):
    #         for index, param_value in enumerate(vr_model.last_updated_param):
    #             if(torch.equal(graph.x[node_indice], param_value)):
    #                 graph.x[node_indice] = vr_model.learnable_param[index].data
    #                 # print(f'匹配成功，第{index}个learnable params')
    #                 # print(f'{graph_list[graph_index].x[node_indice]}\n')
    #                 # print(f'{vr_model.learnable_param[index].data}\n')
    #                 count+=1
    # print(f'耗时间{time.time()-start_time}')
    # import pdb
    # pdb.set_trace()    
    # updated_graph_list = graph_list
    # return updated_graph_list

    # v3使用torch.where函数加快速度 有一点小bug，也就是match_info[0].unique()[-1]好像会匹配上两个不同的点，但是可以确定的是最后一个点和参数值是匹配的。估计是torch.where的bug
    # 该代码用于test文件夹
    count = 0
    for graph_index, graph in enumerate(graph_list):
        # for node_indice in range(graph.num_nodes):
        for index, param_value in enumerate(vr_model.last_updated_param):
            match_info = torch.where(graph.x==param_value)
            if(match_info[0].shape[0]!=0):
                target_node_indice = match_info[0].unique()[-1].item()     # RuntimeError: a Tensor with 2 elements cannot be converted to Scalar
                graph.x[target_node_indice] = vr_model.learnable_param[index].data
                count+=1
    updated_graph_list = graph_list
    return updated_graph_list    

# v2 将metis更换为random walk，但是虚点数量还没有设置为大于1的版本
# 单向边无法进行随机游走！！！
# @param('general.cache_dir')
# @param('pretrain.cross_link')
# @param('pretrain.cl_init_method')
# @param('pretrain.cross_link_ablation')
# @param('pretrain.dynamic_edge')
# @param('pretrain.dynamic_prune')
# @param('pretrain.split_method')
# def get_clustered_data(dataset, cache_dir, num_parts, cross_link, cl_init_method='learnable', cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):

#     from .utils import preprocess, iterate_datasets
#     data_list = [preprocess(data) for data in iterate_datasets(dataset)]
#     from torch_geometric.data import Batch
#     data = Batch.from_data_list(data_list)
#     from copy import deepcopy

#     print(data.num_nodes)
#     vr_model = None
#     # 确定是否要在不同图之间创建连接  
#     if(cross_link > 0):
#         num_graphs = data.num_graphs
#         graph_node_indices = []

#         for graph_index in range(num_graphs):
#             # 找出属于当前图的节点
#             node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
#             graph_node_indices.append(node_indices)
#         # 获取所有Batch graph的点的索引，graph_node_indices列表长度就是Batch的图的数量

#         new_index_list = [i for i in range(data.num_graphs)]
        
#         if(cl_init_method == 'mean'):
#             new_node_features = []
#             # 初始化节点特征，new_index_list是batch graph对象中的图的索引
#             for node_graph_index in new_index_list:
#                 node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#                 new_node_features.append(torch.mean(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

#             new_node_features = torch.cat(new_node_features, dim=0)
#             # 将新加入的节点信息融入源图中
#             data.x = torch.cat([data.x, new_node_features], dim=0)

#         elif(cl_init_method == 'sum'):
#             new_node_features = []
#             # 初始化节点特征，new_index_list是batch graph对象中的图的索引
#             for node_graph_index in new_index_list:
#                 node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#                 new_node_features.append(torch.sum(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

#             new_node_features = torch.cat([new_node_features], dim=0)
#             # 将新加入的节点信息融入源图中
#             data.x = torch.cat([data.x, new_node_features], dim=0)
#         elif(cl_init_method == 'simple'):
#             # 初始化节点特征，为了避免梯度为0，节点特征全部设置为1 
#             new_node_features = torch.ones((len(new_index_list),data.num_node_features))
#             # 将新加入的节点信息融入源图中
#             data.x = torch.cat([data.x, new_node_features], dim=0) 
#         elif(cl_init_method == 'learnable'):
#             from model.virtual_node_model import VirtualNode               
#             vr_model = VirtualNode(data.num_node_features,len(new_index_list)*cross_link) # cross_link=0的时候表示不加虚点和相关的连边，大于0的时候表示每张图添加几个虚点。
#             # 将新加入的节点信息融入源图中
#             data.x = vr_model.add_learnable_features_with_no_grad(data.x)
      
#         # 将节点添加到batch对象中，指定新加入的节点，分别属于哪一张图。
#         data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)

#         if(dynamic_edge=='none'):
#             # 每个virtual node和对应的图，比如cora、wisconsin的节点全连接
#             for node_graph_index in new_index_list:
#                 node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#                 new_node_index = node_indices_corresponding_graph[-1]
                
#                 # 添加单向边
#                 new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
#                 data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                
#                 # 添加双向边
#                 new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
#                 data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            
#             # 如果不执行Ablation Study，virtual nodes之间需要连接，进而将所有的预训练图数据都连通成一个大图
#             if(cross_link_ablation==False):
#                 # virtual nodes之间全连接。注意是双向的即可。
#                 all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
#                 for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
#                     other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index] #双向连接         
#                     # other_added_node_index_list = [index for index in all_added_node_index[list_index+1:]] # 单向链接，和pyg的dataset保持一致
#                     new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
#                     data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
#                 # 把冗余的边删掉，现在是1->2, 2->1，删掉其中一个。
#         elif(dynamic_edge=='internal_external'):
#             all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
#             cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
#             cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
#             cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

#             all_cross_edge = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息
           
#             vr_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index)) #点为虚点则为True，否则为False
#             vr_edge_indices = torch.where(vr_edge_bool_signs)[0] # 找出值为True的边索引值
#             vr_cross_edge=all_cross_edge[:,vr_edge_indices] #筛选出存在virtual nodes的边
#             vr_cross_undirected_edge = torch.sort(vr_cross_edge, dim=0)[0] #根据入点比出点小的原则对shape=(2,xxx)边对象进行排序，转换成无向图边序列
#             vr_cross_undirected_edge_np = vr_cross_undirected_edge.numpy() # 将vr_cross_undirected_edge转换为numpy对象
#             vr_cross_unique_edges = np.unique(vr_cross_undirected_edge_np, axis=1) # 去除序列中冗余的边，保证1 -> 2,而不出现2 -> 1

#             print(f"Added Edge Num is {len(vr_cross_unique_edges[0])}")

#             data.edge_index = torch.cat([data.edge_index, torch.tensor(vr_cross_unique_edges).contiguous()], dim=1)   

#     raw_data = deepcopy(data) #如果加虚点了，就是带虚点的图，否则就是原图

#     if(split_method == 'metis'):
#         import os
#         from torch_geometric.loader.cluster import ClusterData

#         metis_cache_dir = os.path.join(cache_dir, ','.join(dataset))
#         os.makedirs(metis_cache_dir, exist_ok=True)

#         data = list(ClusterData(data, num_parts=num_parts, save_dir=metis_cache_dir))

#         return data, vr_model, raw_data
#     elif(split_method=='RandomWalk'):
#         from torch_cluster import random_walk
#         split_ratio = 0.1
#         walk_length = 30
#         all_random_node_list = torch.randperm(data.num_nodes)
#         selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
#         random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
#         walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

#         graph_list = []  # 存储子图的列表
#         skip_num = 0        
#         for walk in walk_list:   
#             subgraph_nodes = torch.unique(walk)
#             if(len(subgraph_nodes)<5):
#                 skip_num+=1
#                 continue
#             subgraph_data = data.subgraph(subgraph_nodes)

#             graph_list.append(subgraph_data)

#         print(f"Total {len(graph_list)} subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

#         return graph_list, vr_model, raw_data

# 缓存之前的metis的代码 v1
# @param('general.cache_dir')
# @param('pretrain.cross_link')
# @param('pretrain.cl_init_method')
# @param('pretrain.cross_link_ablation')
# @param('pretrain.dynamic_edge')
# @param('pretrain.dynamic_prune')
# def get_clustered_data(dataset, cache_dir, num_parts, cross_link, cl_init_method='learnable', cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0):

#     from .utils import preprocess, iterate_datasets
#     data_list = [preprocess(data) for data in iterate_datasets(dataset)]
#     from torch_geometric.data import Batch
#     data = Batch.from_data_list(data_list)
#     from copy import deepcopy
#     raw_data = deepcopy(data)
#     print(data.num_nodes)
#     vr_model = None
#     # 确定是否要在不同图之间创建连接  
#     if(cross_link==True):
#         num_graphs = data.num_graphs
#         graph_node_indices = []

#         for graph_index in range(num_graphs):
#             # 找出属于当前图的节点
#             node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
#             graph_node_indices.append(node_indices)
#         # 获取所有Batch graph的点的索引，graph_node_indices列表长度就是Batch的图的数量

#         new_index_list = [i for i in range(data.num_graphs)]
        
#         if(cl_init_method == 'mean'):
#             new_node_features = []
#             # 初始化节点特征，new_index_list是batch graph对象中的图的索引
#             for node_graph_index in new_index_list:
#                 node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#                 new_node_features.append(torch.mean(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

#             new_node_features = torch.cat(new_node_features, dim=0)
#             # 将新加入的节点信息融入源图中
#             data.x = torch.cat([data.x, new_node_features], dim=0)

#         elif(cl_init_method == 'sum'):
#             new_node_features = []
#             # 初始化节点特征，new_index_list是batch graph对象中的图的索引
#             for node_graph_index in new_index_list:
#                 node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#                 new_node_features.append(torch.sum(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

#             new_node_features = torch.cat([new_node_features], dim=0)
#             # 将新加入的节点信息融入源图中
#             data.x = torch.cat([data.x, new_node_features], dim=0)
#         elif(cl_init_method == 'simple'):
#             # 初始化节点特征，为了避免梯度为0，节点特征全部设置为1 
#             new_node_features = torch.ones((len(new_index_list),data.num_node_features))
#             # 将新加入的节点信息融入源图中
#             data.x = torch.cat([data.x, new_node_features], dim=0) 
#         elif(cl_init_method == 'learnable'):
#             from model.virtual_node_model import VirtualNode               
#             vr_model = VirtualNode(data.num_node_features,len(new_index_list))
#             # 将新加入的节点信息融入源图中
#             data.x = vr_model.add_learnable_features_with_no_grad(data.x)
#             # data.x = vr_model.add_learnable_features(data.x)
        
#         # print(data.num_nodes)
#         # print(data.edge_index)
#         # print(data.edge_index.shape)
      
#         # 将节点添加到batch对象中，指定新加入的节点，分别属于哪一张图。
#         data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)

#         if(dynamic_edge=='none'):
#             # 每个virtual node和对应的图，比如cora、wisconsin的节点全连接
#             for node_graph_index in new_index_list:
#                 node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#                 new_node_index = node_indices_corresponding_graph[-1]
#                 new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
#                 data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            
#             # 如果不执行Ablation Study，virtual nodes之间需要连接，进而将所有的预训练图数据都连通成一个大图
#             if(cross_link_ablation==False):
#                 # virtual nodes之间全连接。注意是单向的即可。
#                 all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
#                 for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
#                     # other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index] #双向连接         
#                     other_added_node_index_list = [index for index in all_added_node_index[list_index+1:]] # 单向链接，和pyg的dataset保持一致
#                     new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
#                     data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
#                 # 把冗余的边删掉，现在是1->2, 2->1，删掉其中一个。
#         elif(dynamic_edge=='internal_external'):
#             all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
#             cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
#             cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
#             cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

#             all_cross_edge = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息
           
#             vr_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index)) #点为虚点则为True，否则为False
#             vr_edge_indices = torch.where(vr_edge_bool_signs)[0] # 找出值为True的边索引值
#             vr_cross_edge=all_cross_edge[:,vr_edge_indices] #筛选出存在virtual nodes的边
#             vr_cross_undirected_edge = torch.sort(vr_cross_edge, dim=0)[0] #根据入点比出点小的原则对shape=(2,xxx)边对象进行排序，转换成无向图边序列
#             vr_cross_undirected_edge_np = vr_cross_undirected_edge.numpy() # 将vr_cross_undirected_edge转换为numpy对象
#             vr_cross_unique_edges = np.unique(vr_cross_undirected_edge_np, axis=1) # 去除序列中冗余的边，保证1 -> 2,而不出现2 -> 1

#             print(f"Added Edge Num is {len(vr_cross_unique_edges[0])}")

#             data.edge_index = torch.cat([data.edge_index, torch.tensor(vr_cross_unique_edges).contiguous()], dim=1)

#             # 遍历速度慢，其实all_cross_edge根据torch.where()已经可以判断哪些是有virtual node出发的边了
#             # vr_cross_unique_edges = [] #遍历循环，要把根据相似性计算得到的边中由虚拟节点出发的边筛选出来，并且要求是单向的
#             # for virtual_node_index in all_added_node_index:
#             #     all_current_virtual_node_edge_indices = torch.where(all_cross_edge[0]==virtual_node_index)[0].tolist()
#             #     all_current_virtual_node_edge_start_nodes = all_cross_edge[0][all_current_virtual_node_edge_indices]
#             #     all_current_virtual_node_edge_end_nodes = all_cross_edge[1][all_current_virtual_node_edge_indices]
#             #     for edge_id, edge_start_node_id in enumerate(all_current_virtual_node_edge_start_nodes.tolist()): #注意如果tensor长度过长， tolist()函数会直接报错，遍历不了这么多数据
#             #     # 应该修改遍历方式
#             #         edge_end_node_id = all_current_virtual_node_edge_end_nodes[edge_id].item()
#             #         if([edge_end_node_id,edge_start_node_id] in vr_cross_unique_edges):
#             #             continue
#             #         else:
#             #             vr_cross_unique_edges.append([edge_start_node_id,edge_end_node_id])

#             # print(f"Added Edge Num is {len(vr_cross_unique_edges)}")
#             # # import pdb
#             # # pdb.set_trace()

#             # data.edge_index = torch.cat([data.edge_index, torch.tensor(virtual_node_cross_edge).t().contiguous()], dim=1)

#         # print(data.edge_index)
#         # print(data.edge_index.shape)
#         # import pdb
#         # pdb.set_trace()

#     import os
#     from torch_geometric.loader.cluster import ClusterData
    
#     # import pdb
#     # pdb.set_trace()

#     metis_cache_dir = os.path.join(cache_dir, ','.join(dataset))
#     os.makedirs(metis_cache_dir, exist_ok=True)

#     data = list(ClusterData(data, num_parts=num_parts, save_dir=metis_cache_dir))

#     return data, vr_model, raw_data

# @param('pretrain.cross_link_ablation')
# @param('pretrain.dynamic_edge')
# @param('pretrain.dynamic_prune')
# @param('pretrain.split_method')
# def get_clustered_data_for_learnable_vr(data, num_parts, vr_model, cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):

#     # 确定是否要在不同图之间创建连接
#     num_graphs = data.num_graphs
#     graph_node_indices = []

#     for graph_index in range(num_graphs):
#         # 找出属于当前图的节点
#         node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
#         graph_node_indices.append(node_indices)
#     # 获取所有Batch graph的点的索引，graph_node_indices列表长度就是Batch的图的数量

#     new_index_list = [i for i in range(data.num_graphs)]
    
#     # 将新加入的节点信息融入源图中
#     data.x = vr_model.add_learnable_features_with_no_grad(data.x)
#     # data.x = vr_model.add_learnable_features(data.x)
    
#     # 将节点添加到batch对象中，指定新加入的节点，分别属于哪一张图。
#     data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)
#     if(dynamic_edge=='none'):
#         for node_graph_index in new_index_list:
#             node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
#             new_node_index = node_indices_corresponding_graph[-1]
#             new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
#             data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
        
#         # 如果不执行Ablation Study，virtual nodes之间需要连接
#         if(cross_link_ablation==False):
#             # virtual nodes之间全连接。注意是单向的即可。
#             all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
#             for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
#                 # other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index] #双向连接         
#                 other_added_node_index_list = [index for index in all_added_node_index[list_index+1:]] # 单向链接，和pyg的dataset保持一致
#                 new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
#                 data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
#             # 把冗余的边删掉，现在是1->2, 2->1，删掉其中一个。    
#     elif(dynamic_edge=='internal_external'):            
#         all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
#         cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
#         cross_sim = torch.sigmoid(cross_dot)  # 0-1 from prompt to input graph
#         cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

#         all_cross_edge = cross_adj.nonzero().t().contiguous() #nonzero()函数取得时相似性计算矩阵的索引值，[[0,2],[2,3],[2,4]] -> 转置完 [[0,2,2],[2,3,4]]，第二行的索引对应的原图的节点信息
        
#         vr_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index)) #点为虚点则为True，否则为False
#         vr_edge_indices = torch.where(vr_edge_bool_signs)[0] # 找出值为True的边索引值
#         vr_cross_edge=all_cross_edge[:,vr_edge_indices] #筛选出存在virtual nodes的边
#         vr_cross_undirected_edge = torch.sort(vr_cross_edge, dim=0)[0] #根据入点比出点小的原则对shape=(2,xxx)边对象进行排序，转换成无向图边序列
#         vr_cross_undirected_edge_np = vr_cross_undirected_edge.numpy() # 将vr_cross_undirected_edge转换为numpy对象
#         vr_cross_unique_edges = np.unique(vr_cross_undirected_edge_np, axis=1) # 去除序列中冗余的边，保证1 -> 2,而不出现2 -> 1

#         print(f"Added Edge Num is {len(vr_cross_unique_edges[0])}")

#         data.edge_index = torch.cat([data.edge_index, torch.tensor(vr_cross_unique_edges).contiguous()], dim=1)

#     import os
#     from torch_geometric.loader.cluster import ClusterData

#     data = list(ClusterData(data, num_parts=num_parts))         

#     return data