from fastargs.decorators import param
import numpy as np
import torch
from copy import deepcopy

@param('general.save_dir')
@param('data.name', 'dataset')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('pretrain.method')
@param('pretrain.noise_switch')
def run(
    save_dir,
    dataset,
    backbone_model,
    saliency_model,
    method,
    noise_switch,
    ):
     
    if(saliency_model == 'mlp'):    
        # load data
        from data import get_clustered_data
        data = get_clustered_data(dataset)

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
    else:
        # load data
        from data import get_clustered_data

        with torch.no_grad():
            data, gco_model, raw_data = get_clustered_data(dataset) 

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
    
    # train
    if method == 'graphcl':            
        model = graph_cl_pretrain(data, model, gco_model, raw_data)
    elif method == 'simgrace':
        model = simgrace_pretrain(data, model, gco_model, raw_data)
    else:
        raise NotImplementedError(f'Unknown method: {method}')

    # save
    import os

    torch.save(model.state_dict(), os.path.join(save_dir, ','.join(dataset)+'_pretrained_model.pt'))

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
    gco_model,
    raw_data,
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
                                num_workers=4)  
        loader2 = DataLoader(view_list_2, batch_size=batch_size, shuffle=False,
                                num_workers=4)  

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    
    loss_fn = ContrastiveLoss(model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = 100000.
    best_model = None
    if(gco_model==None):
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
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from data.contrastive import update_graph_list_param
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)

            loaders = get_loaders(data)
        elif(e==0):
            loaders = get_loaders(data)

        pbar = tqdm(zip(*loaders), total=len(loaders[0]), ncols=100, desc=f'Epoch {e}, Loss: inf')
                
        for batch1, batch2 in pbar:

            if(gco_model!=None):
                batch1 = gco_model(batch1)
                batch2 = gco_model(batch2)    

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

        if(gco_model!=None):
            data  = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model

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
    gco_model,
    raw_data,
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


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    

    loss_fn = SimgraceLoss(model.backbone, model.backbone.hidden_dim).to(device)
    loss_fn.train(), model.to(device).train()
    best_loss = np.inf
    best_model = None
    if(gco_model==None):
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
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )
        else:
            rec_loss_fn = ReconstructionLoss(model.backbone.hidden_dim, data[0].num_node_features).to(device)
            rec_loss_fn.train()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(gco_model.parameters()) + list(model.parameters()) + list(loss_fn.parameters()) +list(rec_loss_fn.parameters())),
                lr=learning_rate,
                weight_decay=weight_decay
                )            

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)

    from torchmetrics import MeanMetric
    from tqdm import tqdm
    from data.contrastive import update_graph_list_param
    loss_metric = MeanMetric()

    for e in range(epoch):
        
        loss_metric.reset()

        if(cross_link > 0 and cl_init_method == 'learnable'):
            if(split_method=='RandomWalk'):
                last_updated_data = deepcopy(data)

            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) 
        elif(e==0):
            loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1) 

        pbar = tqdm(loader, total=len(loader), ncols=100, desc=f'Epoch {e}, Loss: inf')

        for batch1 in pbar:
            if(gco_model!=None):
                batch1 = gco_model(batch1)

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

        if(gco_model!=None):
            data  = update_graph_list_param(last_updated_data, gco_model)
            gco_model.update_last_params()

        # lr_scheduler.step()
        
        if(loss_metric.compute()<best_loss):
            best_loss = loss_metric.compute()
            best_model = deepcopy(model)
            
        pbar.close()
        
    return best_model
    