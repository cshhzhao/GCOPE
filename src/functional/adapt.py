import copy
from fastargs.decorators import param
import torch


@param('data.name', 'dataset')
@param('adapt.batch_size')
@param('data.supervised.ratios')
@param('adapt.method')
@param('model.backbone.model_type', 'backbone_model')
@param('model.saliency.model_type', 'saliency_model')
@param('model.answering.model_type', 'answering_model')
@param('adapt.pretrained_file')
@param('general.save_dir')
@param('adapt.repeat_times')
def run(
    dataset,
    batch_size,
    ratios,
    method,
    backbone_model,
    saliency_model,
    answering_model,
    pretrained_file,
    save_dir,
    repeat_times,
    ):
    
    # load data
    from data import get_supervised_data
    from torch_geometric.loader import DataLoader
    datasets, num_classes = get_supervised_data(dataset[0], ratios=ratios)
    loaders = { k: DataLoader(v, batch_size=batch_size, shuffle=True, num_workers=4) for k, v in datasets.items() }

    # init model
    from model import get_model
    model = get_model(
        backbone_kwargs = {
            'name': backbone_model,
            'num_features': datasets['train'][0].x.size(-1),
        },
        answering_kwargs = {
            'name': answering_model,
            'num_class': num_classes,
        },
        saliency_kwargs = {
            'name': saliency_model,
            'feature_dim': datasets['train'][0].x.size(-1),
        } if saliency_model != 'none' else None,
    )

    model.load_state_dict(torch.load(pretrained_file,map_location=lambda storage, loc: storage.cuda(1)), strict=False)

    # train
    all_results = []
    for _ in range(repeat_times):
        if method == 'finetune':
            results = finetune(loaders, model)
        elif method == 'prog':
            from model import get_prompt_model
            # statistic the average node number of dataset
            total_graph = sum([len(v) for k, v in datasets.items()])
            train_node_num = sum([g.num_nodes for g in datasets['train']])
            val_node_num = sum([g.num_nodes for g in datasets['val']])
            test_node_num = sum([g.num_nodes for g in datasets['test']])
            prompt_node_num = int((train_node_num + val_node_num + test_node_num) / total_graph)
            prompt_model = get_prompt_model(num_features=datasets['train'][0].x.size(-1), prompt_node_num=prompt_node_num)
            results = prog(loaders, model, prompt_model, dataset)        
        else:
            raise NotImplementedError(f'Unknown method: {method}')
        
        results.pop('model')
        all_results.append(results)        

    # print acc, auroc, f1 with std
    import numpy as np
    for k in all_results[0].keys():
        print(f'{k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}')
        
    import os

    if(method!='prog'):
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(method+f'FT on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')
    else:
        with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
            f.write('-------------------------------------------------\n')
            for k in all_results[0].keys():
                f.write(method+f' on All, Target Dataset: {dataset[0]}, {k}: {np.mean([r[k] for r in all_results]):.4f} ± {np.std([r[k] for r in all_results]):.4f}\n')                        
            
    # save
    # torch.save(results, os.path.join(save_dir, dataset[0]+'_results.pt'))

@param('adapt.finetune.backbone_tuning')
@param('adapt.finetune.saliency_tuning')
@param('adapt.finetune.learning_rate')
@param('adapt.finetune.weight_decay')
@param('adapt.epoch')
def finetune(
        loaders,
        model,
        backbone_tuning,
        saliency_tuning,
        learning_rate,
        weight_decay,
        epoch,
        ):

    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = learning_rate,
        weight_decay = weight_decay,
        )

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)

    best_acc = 0.
    best_model = None

    for e in range(epoch):
        model.train()

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()

        pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} Training, Loss: inf')

        for batch in pbar:
            optimizer.zero_grad()
            batch = batch.to(device)
            pred = model(batch)
            loss = torch.nn.functional.cross_entropy(pred, batch.y)
            loss.backward()
            optimizer.step()

            loss_metric.update(loss.detach(), batch.size(0))
            pbar.set_description(f'Epoch {e} Training Loss: {loss_metric.compute():.4f}', refresh=True)
        pbar.close()

        model.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:
                batch = batch.to(device)
                pred = model(batch).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(batch), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_model = deepcopy(model)
    
    model = best_model if best_model is not None else model

    # test
    model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(batch), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()
    
    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }

@param('adapt.epoch')
@param('adapt.prog.prompt_lr')
@param('adapt.prog.prompt_weight_decay')
@param('adapt.prog.ans_lr')
@param('adapt.prog.ans_weight_decay')
@param('adapt.prog.backbone_tuning')
@param('adapt.prog.saliency_tuning')
def prog(
        loaders,
        model,
        prompt_model,      
        dataset,
        epoch,
        backbone_tuning,
        saliency_tuning,          
        prompt_lr,
        prompt_weight_decay,
        ans_lr,
        ans_weight_decay,
        ):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.backbone.to(device)
    model.answering.to(device)
    prompt_model.to(device)
    
    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)

    opi_pg = torch.optim.Adam(
        prompt_model.parameters(),
        lr = prompt_lr,
        weight_decay = prompt_weight_decay,
        )
    
    opi_answer = torch.optim.Adam(
        model.answering.parameters(),
        lr = ans_lr,
        weight_decay = ans_weight_decay,
        )    

    from torchmetrics import MeanMetric, Accuracy, F1Score, AUROC
    from tqdm import tqdm
    from copy import deepcopy

    loss_metric = MeanMetric().to(device)
    acc_metric = Accuracy(task='multiclass', num_classes=model.answering.num_class).to(device)
    f1_metric = F1Score(task='multiclass', num_classes=model.answering.num_class, average='macro').to(device)
    auroc_metric = AUROC(task="multiclass", num_classes=model.answering.num_class).to(device)
    
    # load prompting data

    from torch_geometric.loader import DataLoader

    best_acc = 0.
    best_backbone = None
    best_prompt_model = None
    best_answering = None

    for e in range(epoch):

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()
        
        print(("{}/{} frozen gnn | *tune prompt and tune answering function...".format(e, epoch)))
        prompt_model.train()
        model.backbone.eval()
        model.answering.train()

        from tqdm import tqdm

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        running_loss = 0.
        
        ans_pbar = tqdm(loaders['train'], total=len(loaders['train']), ncols=100, desc=f'Epoch {e} / Total Epoch {epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):  # bar2       
            
            train_batch = train_batch.to(device)
            prompted_graph = prompt_model(train_batch)

            graph_emb = model.backbone(prompted_graph)

            # print(graph_emb)
            pred = model.answering(graph_emb)
            # print(pre)
            train_loss = torch.nn.functional.cross_entropy(pred, train_batch.y)

            opi_answer.zero_grad()
            opi_pg.zero_grad()
            train_loss.backward()
            opi_answer.step()
            opi_pg.step()
            running_loss += train_loss.item()

            current_avg_last_loss = running_loss / (batch_id+1)  # loss per batch

            ans_pbar.set_description('Epoch {} / Total Epoch {} | avg loss: {:.8f}'.format(e, epoch, current_avg_last_loss), refresh=True)
        
        ans_pbar.close()        
                
        model.backbone.eval()
        prompt_model.eval()
        model.answering.eval()

        pbar = tqdm(loaders['val'], total=len(loaders['val']), ncols=100, desc=f'Epoch {e} Validation, Acc: 0., F1: 0.')
        with torch.no_grad():
            for batch in pbar:              
                batch = batch.to(device)
                prompted_graph = prompt_model(batch)
                z = model.backbone(prompted_graph)
                pred = model.answering(z).argmax(dim=-1)

                acc_metric.update(pred, batch.y)
                f1_metric.update(pred, batch.y)
                auroc_metric.update(model(prompted_graph), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_backbone = deepcopy(model.backbone)
            best_answering = deepcopy(model.answering)
            best_prompt_model = deepcopy(prompt_model)
    
    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering
    prompt_model = best_prompt_model if best_prompt_model is not None else prompt_model

    # test
    model.backbone.eval()
    model.answering.eval()
    prompt_model.eval()

    acc_metric.reset()
    f1_metric.reset()
    auroc_metric.reset()

    pbar = tqdm(loaders['test'], total=len(loaders['test']), ncols=100, desc=f'Testing, Acc: 0., F1: 0.')
    with torch.no_grad():
        for batch in pbar:
            batch = batch.to(device)
            prompted_graph = prompt_model(batch)
            z = model.backbone(prompted_graph)
            pred = model.answering(z).argmax(dim=-1)

            acc_metric.update(pred, batch.y)
            f1_metric.update(pred, batch.y)
            auroc_metric.update(model(prompted_graph), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }