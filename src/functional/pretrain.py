from fastargs.decorators import param
import numpy as np
import torch
from copy import deepcopy
# from model.saliency.featurizer import FeatureTokenizer

# 这份代码缺点是：每个epoch中使用的random walk的subgraphs是一套，样本多样性有区别
# 优点是每个epoch中使用的random walk的subgraphs是一套，每个epoch间隔的数据预处理时间短。
# 每个batch中subgraph的特征都是上一个epoch更新后的节点特征，但是更新上去的都是每一个batch执行后最新的虚点特征。

@param('general.save_dir')
@param('data.name', 'dataset')
@param('data.clustered.num_parts')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('pretrain.method')
@param('pretrain.noise_switch')
def run(
    save_dir,
    dataset,
    num_parts,
    backbone_model,
    saliency_model,
    method,
    noise_switch,
    ):
     
    if(saliency_model == 'mlp'):    
        # load data
        from data import get_clustered_data
        data = get_clustered_data(dataset, num_parts=num_parts)

        # init model
        from model import get_model
        model = get_model(
            backbone_kwargs = {
                'name': backbone_model,
                'num_features': data[0].x.size(-1),
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_dim': data[0].x.size(-1),
            } if saliency_model != 'none' else None,
        )
    elif(saliency_model == 'transformer'):
        # load data
        from data import get_clustered_data
        data,feature_to_tokens = get_clustered_data(dataset, num_parts=num_parts)
        
        # init model
        from model import get_model
        model = get_model(
            backbone_kwargs = {
                'name': backbone_model,
                'num_features': data[0].x.size(-1),
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_to_tokens': feature_to_tokens,
            } if saliency_model != 'none' else None,
        )        
    else:
        # load data
        from data import get_clustered_data

        with torch.no_grad():
            # data,feature_to_tokens = get_clustered_data(dataset, num_parts=num_parts)
            # vr_model(virtual_node_model)只有cl_init_method (cross_link_initialization_method) = 'learnable'的时候才会返回模型，其他情况下都是none
            data, vr_model, raw_data = get_clustered_data(dataset, num_parts=num_parts) 

        # init model
        from model import get_model
        model = get_model(
            backbone_kwargs = {
                'name': backbone_model,
                'num_features': data[0].x.size(-1),
            },
            saliency_kwargs = {
                'name': saliency_model,
                'feature_to_tokens': feature_to_tokens,
            } if saliency_model != 'none' else None,
        )                
    
    # train
    if(noise_switch == False):
        if method == 'graphcl':            
            model = graph_cl_pretrain(data, model, vr_model, raw_data, num_parts) #后面两个参数，只要在virtual nodes是可学习的情况下才有用
        elif method == 'simgrace':
            model = simgrace_pretrain(data, model, vr_model, raw_data, num_parts) #后面两个参数，只要在virtual nodes是可学习的情况下才有用            
        else:
            raise NotImplementedError(f'Unknown method: {method}')
    else:
        if method == 'graphcl':
            model = graph_cl_pretrain_with_noise(data, model, vr_model, raw_data, num_parts) #后面两个参数，只要在virtual nodes是可学习的情况下才有用
        else:
            raise NotImplementedError(f'Unknown method: {method}')        
        
    # save
    import os
    # torch.save(model.state_dict(), os.path.join(save_dir, dataset+'_pretrained_model.pt'))

    # import pdb
    # pdb.set_trace()    
    torch.save(model.state_dict(), os.path.join(save_dir, ','.join(dataset)+'_pretrained_model.pt'))


# 手动调整参数的代码
@param('pretrain.learning_rate')
@param('pretrain.weight_decay')
@param('pretrain.epoch')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('general.reconstruct')
@param('pretrain.split_method')
@param('pretrain.dynamic_edge')
def graph_cl_pretrain(
    data,
    model,
    vr_model,
    raw_data,
    num_parts,
    learning_rate,
    weight_decay,
    epoch,
    cross_link,
    cl_init_method,
    reconstruct,
    dynamic_edge,
    split_method,
    ):
    
    @param('pretrain.batch_size')
    def get_loaders(data, batch_size):

        import random
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from algorithm.graph_augment import graph_views

        augs, aug_ratio = random.choices(['dropN', 'permE', 'maskN'], k=2), random.randint(1, 3) * 1.0 / 10

        view_list_1 = []
        view_list_2 = []
        for g in data:
            view_g = graph_views(data=g, aug=augs[0], aug_ratio=aug_ratio)
            view_list_1.append(Data(x=view_g.x, edge_index=view_g.edge_index))
            view_g = graph_views(data=g, aug=augs[1], aug_ratio=aug_ratio)
            view_list_2.append(Data(x=view_g.x, edge_index=view_g.edge_index))

        loader1 = DataLoader(view_list_1, batch_size=batch_size, shuffle=False,
                                num_workers=4)  # you must set shuffle=False !
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=4)  # you must set shuffle=False !

        return loader1, loader2

    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, hidden_dim, temperature=0.5):
            super(ContrastiveLoss, self).__init__()
            self.head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature

        def forward(self, zi, zj):
            batch_size = zi.size(0)
            x1_abs = zi.norm(dim=1)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features):
            reconstruction_features = self.decoder(hidden_features)
            return self.loss_fn(input_features, reconstruction_features)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   
    
    # 注意这里是生成对比学习数据，会修改data对象中的节点和边的数量，导致后面更新参数的时候会出现问题
    # loaders = get_loaders(data) #每一次都要重新划分一次数据

    loss_fn = ContrastiveLoss(model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = 100000.
    best_model = None
    if(vr_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(vr_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(vr_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from data.contrastive import get_clustered_data_for_learnable_vr, update_graph_list_param
    loss_metric = MeanMetric()

    # if(vr_model!=None):
    #     last_updated_data = deepcopy(data)

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='metis'):
                data = get_clustered_data_for_learnable_vr(deepcopy(raw_data), num_parts, vr_model)
            elif(split_method=='RandomWalk'):
                # 因为经过get_loaders()函数之后，当前的data数据会经过处理生成对比学习的图样本对，导致一些data中的点和边丢失，因为要提前备份data数据至last_updated_data对象中
                last_updated_data = deepcopy(data)

            loaders = get_loaders(data) #每一次都要重新划分一次数据
        elif(e==0):
            # 非cross_link的情况下，loaders对象只需要实例化一次，否则会导致数据中的节点、边多次随机删除
            loaders = get_loaders(data) #每一次都要重新划分一次数据

        pbar = tqdm(zip(*loaders), total=len(loaders[0]), ncols=100, desc=f'Epoch {e}, Loss: inf')
                
        for batch1, batch2 in pbar:
            # 需要调整，每一个batch里面，更新完之后，没办法再匹配数据了？
            if(vr_model!=None):
                batch1 = vr_model(batch1)
                batch2 = vr_model(batch2)    

            # import pdb
            # pdb.set_trace()

            optimizer.zero_grad()

            if(reconstruct==0.0):
                zi, zj = model(batch1.to(device)), model(batch2.to(device))
                loss = loss_fn(zi, zj)
            else:               
                zi, hi = model(batch1.to(device))
                zj, hj = model(batch2.to(device))
                loss = loss_fn(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch2.x, hj))
                
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), batch1.size(0))
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)

        # 更新vr_model里面的last_updated_param参数。   
        if(vr_model!=None):
            # 注意下面两行代码的调用顺序
            data  = update_graph_list_param(last_updated_data, vr_model)
            vr_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model

        # print(vr_model.learnable_param[0][-10:])
        # if(e==10):
        #     import pdb
        #     pdb.set_trace()

@param('pretrain.learning_rate')
@param('pretrain.weight_decay')
@param('pretrain.epoch')
@param('pretrain.cross_link')
@param('pretrain.cl_init_method')
@param('general.reconstruct')
@param('pretrain.split_method')
@param('pretrain.dynamic_edge')
@param('pretrain.batch_size')
def simgrace_pretrain(
    data,
    model,
    vr_model,
    raw_data,
    num_parts,
    learning_rate,
    weight_decay,
    epoch,
    cross_link,
    cl_init_method,
    reconstruct,
    dynamic_edge,
    split_method,
    batch_size,
    ):

    from torch_geometric.loader import DataLoader
    from data import gen_ran_output

    class SimgraceLoss(torch.nn.Module):
        def __init__(self, gnn, hidden_dim, temperature=0.5):
            super(SimgraceLoss, self).__init__()
            self.gnn = gnn
            self.projection_head = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.temperature = temperature

        # 用于simgrace预训练
        
        @param('general.reconstruct')
        def forward_cl(self, data, reconstruct):
        
            if(reconstruct==0.0):
                zi = self.gnn(data)
                zi = self.projection_head(zi)

                return zi
            else:
                zi, hi = self.gnn(data)
                zi = self.projection_head(zi)
            
                return zi, hi
        
        # 计算的是loss_cl()
        def loss_cl(self, zi, zj):
            batch_size = zi.size(0)
            x1_abs = zi.norm(dim=1)
            x2_abs = zj.norm(dim=1)
            sim_matrix = torch.einsum('ik,jk->ij', zi, zj) / torch.einsum('i,j->ij', x1_abs, x2_abs)
            sim_matrix = torch.exp(sim_matrix / self.temperature)
            pos_sim = sim_matrix[range(batch_size), range(batch_size)]
            loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
            loss = - torch.log(loss).mean()
            return loss

    class ReconstructionLoss(torch.nn.Module):
        def __init__(self, hidden_dim, feature_num):
            super(ReconstructionLoss, self).__init__()
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(hidden_dim, feature_num),
            )

            self.loss_fn = torch.nn.MSELoss()

        def forward(self, input_features, hidden_features):
            reconstruction_features = self.decoder(hidden_features)
            return self.loss_fn(input_features, reconstruction_features)

    # import pdb
    # pdb.set_trace()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   
    
    # 注意这里是生成对比学习数据，会修改data对象中的节点和边的数量，导致后面更新参数的时候会出现问题
    # loaders = get_loaders(data) #每一次都要重新划分一次数据

    loss_fn = SimgraceLoss(model.backbone, model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = np.inf
    best_model = None
    if(vr_model==None):
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )                
    else:
        if(reconstruct==0.0):
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(vr_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(vr_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from data.contrastive import get_clustered_data_for_learnable_vr, update_graph_list_param
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='metis'):
                data = get_clustered_data_for_learnable_vr(deepcopy(raw_data), num_parts, vr_model)
            elif(split_method=='RandomWalk'):
                # 因为经过get_loaders()函数之后，当前的data数据会经过处理生成对比学习的图样本对，导致一些data中的点和边丢失，因为要提前备份data数据至last_updated_data对象中
                last_updated_data = deepcopy(data)

            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) #每一次都要重新划分一次数据
        elif(e==0):
            # 非cross_link的情况下，loaders对象只需要实例化一次，否则会导致数据中的节点、边多次随机删除
            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) #每一次都要重新划分一次数据

        pbar = tqdm(loader, total=len(loader), ncols=100, desc=f'Epoch {e}, Loss: inf')

        for batch1 in pbar:
            # 需要调整，每一个batch里面，更新完之后，没办法再匹配数据了？
            if(vr_model!=None):
                batch1 = vr_model(batch1)

            # import pdb
            # pdb.set_trace()

            optimizer.zero_grad()

            if(reconstruct==0.0):
                batch1 = batch1.to(device)
                zi = loss_fn.forward_cl(batch1)                
                zj = gen_ran_output(batch1, loss_fn)
                zj = zj.detach().data.to(device)
                loss = loss_fn.loss_cl(zi, zj)              
            else:

                batch1 = batch1.to(device)
                zi, hi = loss_fn.forward_cl(batch1)
                zj, hj = gen_ran_output(batch1, loss_fn)
                zj = zj.detach().data.to(device)
                loss = loss_fn.loss_cl(zi, zj) + reconstruct*(rec_loss_fn(batch1.x, hi) + rec_loss_fn(batch1.x, hj))
                
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), batch1.size(0))
            pbar.set_description(f'Epoch {e}, Loss {loss_metric.compute():.4f}', refresh=True)

        # 更新vr_model里面的last_updated_param参数。
        if(vr_model!=None):
            # 注意下面两行代码的调用顺序
            data  = update_graph_list_param(last_updated_data, vr_model)
            vr_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model
    