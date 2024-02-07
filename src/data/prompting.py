from fastargs.decorators import param
import torch


def induced_graphs(data, smallest_size=10, largest_size=30):

    from torch_geometric.utils import subgraph, k_hop_subgraph
    from torch_geometric.data import Data
    import numpy as np

    induced_graph_list = []

    for index in range(data.x.size(0)):
        current_label = data.y[index].item()

        current_hop = 2
        subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                            edge_index=data.edge_index, relabel_nodes=True)
        
        while len(subset) < smallest_size and current_hop < 5:
            current_hop += 1
            subset, _, _, _ = k_hop_subgraph(node_idx=index, num_hops=current_hop,
                                                edge_index=data.edge_index)
            
        if len(subset) < smallest_size:
            need_node_num = smallest_size - len(subset)
            pos_nodes = torch.argwhere(data.y == int(current_label)) 
            candidate_nodes = torch.from_numpy(np.setdiff1d(pos_nodes.numpy(), subset.numpy()))
            candidate_nodes = candidate_nodes[torch.randperm(candidate_nodes.shape[0])][0:need_node_num]
            subset = torch.cat([torch.flatten(subset), torch.flatten(candidate_nodes)])

        if len(subset) > largest_size:
            subset = subset[torch.randperm(subset.shape[0])][0:largest_size - 1]
            subset = torch.unique(torch.cat([torch.LongTensor([index]), torch.flatten(subset)]))

        sub_edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True)

        x = data.x[subset]

        induced_graph = Data(x=x, edge_index=sub_edge_index, y=current_label)
        induced_graph_list.append(induced_graph)

    return induced_graph_list

@param('data.clustered.num_parts')
def get_unsupervised_prompting_data(target_dataset, source_dataset, num_parts):
    from .utils import preprocess, iterate_datasets

    target_data = preprocess(next(iterate_datasets(target_dataset)))
    target_graph_list = induced_graphs(target_data)
       
    source_data_list = get_clustered_data(source_dataset, num_parts)

    return {'source':source_data_list,
            'target':target_graph_list}


@param('general.cache_dir')
@param('adapt.prompt.cross_link')
def get_clustered_data(dataset, num_parts, cache_dir, cross_link):

    from .utils import preprocess, iterate_datasets
    data_list = [preprocess(data) for data in iterate_datasets(dataset)]

    from torch_geometric.data import Batch
    data = Batch.from_data_list(data_list)

    # 确定是否要在不同图之间创建连接
    if(cross_link > 0):
        num_graphs = data.num_graphs
        graph_node_indices = []

        for graph_index in range(num_graphs):
            # 找出属于当前图的节点
            node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_node_indices.append(node_indices)
        # 获取所有Batch graph的点的索引，graph_node_indices列表长度就是Batch的图的数量

        new_index_list = [i for i in range(data.num_graphs)]
        
        # 初始化节点特征，为了避免梯度为0，节点特征全部设置为1 
        new_node_features = torch.ones((len(new_index_list),data.num_node_features))

        # 将新加入的节点信息融入源图中
        data.x = torch.cat([data.x, new_node_features], dim=0)
        
        # 将节点添加到batch对象中，指定新加入的节点，分别属于哪一张图。
        data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)
        for node_graph_index in new_index_list:
            node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
            new_node_index = node_indices_corresponding_graph[-1]
            new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
            data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
        
        all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
        for new_node_index in all_added_node_index:
            other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index]               
            new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
            data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)              

    import os
    from torch_geometric.loader.cluster import ClusterData
    metis_cache_dir = os.path.join(cache_dir, ','.join(dataset))
    os.makedirs(metis_cache_dir, exist_ok=True)

    # import pdb
    # pdb.set_trace()    
    data = list(ClusterData(data, num_parts=num_parts, save_dir=metis_cache_dir))

    return data
