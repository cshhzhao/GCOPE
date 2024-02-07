import time
from fastargs.decorators import param
import torch
from torch_geometric.data import Batch
import numpy as np

@param('general.cache_dir')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('pretrain.cross_link_ablation')
@param('pretrain.dynamic_edge')
@param('pretrain.dynamic_prune')
@param('pretrain.split_method')
def get_clustered_data(dataset, cache_dir, cross_link, cl_init_method='learnable', cross_link_ablation=False, dynamic_edge='none',dynamic_prune=0.0,split_method='RandomWalk'):

    from .utils import preprocess, iterate_datasets
    data_list = [preprocess(data) for data in iterate_datasets(dataset)]
    from torch_geometric.data import Batch
    data = Batch.from_data_list(data_list)
    from copy import deepcopy

    data_for_similarity_computation = deepcopy(data)

    print(f'Isolated graphs have total {data.num_nodes} nodes, each dataset added {cross_link} graph coordinators')
    gco_model = None

    if(cross_link > 0):
        num_graphs = data.num_graphs
        graph_node_indices = []

        for graph_index in range(num_graphs):
            node_indices = (data.batch == graph_index).nonzero(as_tuple=False).view(-1)
            graph_node_indices.append(node_indices)

        new_index_list = [i for i in range(num_graphs)]*cross_link 
        
        if(cl_init_method == 'mean'):
            new_node_features = []
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                new_node_features.append(torch.mean(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

            new_node_features = torch.cat(new_node_features, dim=0)
            data.x = torch.cat([data.x, new_node_features], dim=0)

        elif(cl_init_method == 'sum'):
            new_node_features = []
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                new_node_features.append(torch.sum(data.x[node_indices_corresponding_graph],dim=0).reshape((1,-1)))

            new_node_features = torch.cat([new_node_features], dim=0)
            data.x = torch.cat([data.x, new_node_features], dim=0)
        elif(cl_init_method == 'simple'):
            new_node_features = torch.ones((len(new_index_list),data.num_node_features))
            data.x = torch.cat([data.x, new_node_features], dim=0) 
        elif(cl_init_method == 'learnable'):
            from model.graph_coordinator import GraphCoordinator               
            gco_model = GraphCoordinator(data.num_node_features,len(new_index_list))
            data.x = gco_model.add_learnable_features_with_no_grad(data.x)
      
        data.batch = torch.cat([data.batch, torch.tensor([new_index_list]).squeeze(0)], dim=0)

        if(dynamic_edge=='none'):
            if(cross_link==1):            
                for node_graph_index in new_index_list:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index = node_indices_corresponding_graph[-1]
                    
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)                                    
            
            if(cross_link_ablation==False):
                all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]
                for list_index, new_node_index in enumerate(all_added_node_index[:-1]):
                    other_added_node_index_list = [index for index in all_added_node_index if index != new_node_index]
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), torch.tensor(other_added_node_index_list))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
        elif(dynamic_edge=='internal_external'):
            all_added_node_index = [i for i in range(data.num_nodes-len(new_index_list),data.num_nodes)]         
            cross_dot = torch.mm(data.x, torch.transpose(data.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot) 
            cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

            all_cross_edge = cross_adj.nonzero().t().contiguous() 
           
            gco_edge_bool_signs = torch.isin(all_cross_edge[0], torch.tensor(all_added_node_index))
            gco_edge_indices = torch.where(gco_edge_bool_signs)[0]
            gco_cross_edge=all_cross_edge[:,gco_edge_indices]
            gco_cross_undirected_edge = torch.sort(gco_cross_edge, dim=0)[0]
            gco_cross_undirected_edge_np = gco_cross_undirected_edge.numpy()
            gco_cross_unique_edges = np.unique(gco_cross_undirected_edge_np, axis=1)

            print(f"Added Edge Num is {len(gco_cross_unique_edges[0])}")

            data.edge_index = torch.cat([data.edge_index, torch.tensor(gco_cross_unique_edges).contiguous()], dim=1)   
        elif(dynamic_edge=='similarity'):

            if(cross_link==1):            
                for node_graph_index in new_index_list:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index = node_indices_corresponding_graph[-1]
                    
                    new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1])
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                    
                    new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1], torch.tensor([new_node_index]))
                    data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
            else:
                for node_graph_index in new_index_list[:num_graphs]:
                    node_indices_corresponding_graph = (data.batch == node_graph_index).nonzero(as_tuple=False).view(-1)
                    new_node_index_list = node_indices_corresponding_graph[-1*cross_link:]
                    for new_node_index in new_node_index_list:
                        new_edges = torch.cartesian_prod(torch.tensor([new_node_index]), node_indices_corresponding_graph[:-1*cross_link])
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)
                        
                        new_edges = torch.cartesian_prod(node_indices_corresponding_graph[:-1*cross_link], torch.tensor([new_node_index]))
                        data.edge_index = torch.cat([data.edge_index, new_edges.t()], dim=1)     
                                    
            graph_mean_features = []
            for node_graph_index in new_index_list:
                node_indices_corresponding_graph_for_simi = (data_for_similarity_computation.batch == node_graph_index).nonzero(as_tuple=False).view(-1)              
                graph_mean_features.append(torch.mean(data_for_similarity_computation.x[node_indices_corresponding_graph_for_simi],dim=0).tolist())

            graph_mean_features = torch.tensor(graph_mean_features)
            cross_dot = torch.mm(graph_mean_features, torch.transpose(graph_mean_features, 0, 1))
            cross_sim = torch.sigmoid(cross_dot) 
            cross_adj = torch.where(cross_sim < dynamic_prune, 0, cross_sim)

            all_cross_edge = cross_adj.nonzero().t().contiguous()
            all_cross_edge += data_for_similarity_computation.num_nodes
            
            total_edge_num_before_add_cross_link_internal_edges = data.edge_index.shape[1]
            data.edge_index = torch.cat([data.edge_index, all_cross_edge], dim=1)               
            if((data.edge_index.shape[1]-total_edge_num_before_add_cross_link_internal_edges)==all_cross_edge.shape[1]):
                print(f'Edge num after gco connected together{data.edge_index.shape[1]}, totally add {all_cross_edge.shape[1]} inter-dataset edges')

    print(f'Unified graph has {data.num_nodes} nodes, each graph includes {cross_link} graph coordinators')

    raw_data = deepcopy(data)

    if(split_method=='RandomWalk'):
        from torch_cluster import random_walk
        split_ratio = 0.1
        walk_length = 30
        all_random_node_list = torch.randperm(data.num_nodes)
        selected_node_num_for_random_walk = int(split_ratio * data.num_nodes)
        random_node_list = all_random_node_list[:selected_node_num_for_random_walk]
        walk_list = random_walk(data.edge_index[0], data.edge_index[1], random_node_list, walk_length=walk_length)

        graph_list = [] 
        skip_num = 0        
        for walk in walk_list:   
            subgraph_nodes = torch.unique(walk)
            if(len(subgraph_nodes)<5):
                skip_num+=1
                continue
            subgraph_data = data.subgraph(subgraph_nodes)

            graph_list.append(subgraph_data)

        print(f"Total {len(graph_list)} subgraphs with nodes more than 5, and there are {skip_num} skipped subgraphs with nodes less than 5.")

        return graph_list, gco_model, raw_data

def update_graph_list_param(graph_list, gco_model):
    
    count = 0
    for graph_index, graph in enumerate(graph_list):
        for index, param_value in enumerate(gco_model.last_updated_param):
            match_info = torch.where(graph.x==param_value)
            if(match_info[0].shape[0]!=0):
                target_node_indice = match_info[0].unique()[-1].item()
                graph.x[target_node_indice] = gco_model.learnable_param[index].data
                count+=1
    updated_graph_list = graph_list
    return updated_graph_list    