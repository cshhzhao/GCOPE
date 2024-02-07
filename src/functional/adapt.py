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
    # model.load_state_dict(torch.load(pretrained_file), strict=False) #cpu内存不够
    model.load_state_dict(torch.load(pretrained_file,map_location=lambda storage, loc: storage.cuda(1)), strict=False)

    # train
    all_results = []
    for _ in range(repeat_times):
        if method == 'finetune':
            results = finetune(loaders, model)
        elif method == 'prompt':
            from model import get_prompt_model
            # statistic the average node number of dataset
            total_graph = sum([len(v) for k, v in datasets.items()])
            train_node_num = sum([g.num_nodes for g in datasets['train']])
            val_node_num = sum([g.num_nodes for g in datasets['val']])
            test_node_num = sum([g.num_nodes for g in datasets['test']])
            prompt_node_num = int((train_node_num + val_node_num + test_node_num) / total_graph)    
            prompt_model = get_prompt_model(num_features=datasets['train'][0].x.size(-1), prompt_node_num=prompt_node_num)        
            results = prompt(loaders, model, prompt_model, dataset)
        elif method == 'prog':
            from model import get_prompt_model
            # statistic the average node number of dataset
            total_graph = sum([len(v) for k, v in datasets.items()])
            train_node_num = sum([g.num_nodes for g in datasets['train']])
            val_node_num = sum([g.num_nodes for g in datasets['val']])
            test_node_num = sum([g.num_nodes for g in datasets['test']])
            prompt_node_num = int((train_node_num + val_node_num + test_node_num) / total_graph)
            # prompt_node_num = 10
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
        
    # log
    import os
    # with open(os.path.join(save_dir, dataset+'_results.txt'), 'w') as f:
    #     f.write(f'Acc: {results["acc"]:.4f}, F1: {results["f1"]:.4f}\n')
    
    # with open(os.path.join(save_dir, dataset[0]+'_results.txt'), 'a+') as f:
    #     f.write(method+f'FT on All, Target Dataset: {dataset[0]}, Acc: {results["acc"]:.4f}, Auroc: {results["auroc"]:.4f}, F1: {results["f1"]:.4f}\n')

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

    # import random
    # import numpy as np
    # import torch
    # seed = np.random.randint(0, 100000)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    model.backbone.requires_grad_(backbone_tuning)
    model.saliency.requires_grad_(saliency_tuning)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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
@param('adapt.prompt.prompt_lr')
@param('adapt.prompt.prompt_weight_decay')
@param('adapt.prompt.ans_lr')
@param('adapt.prompt.ans_weight_decay')
@param('adapt.prompt.source_dataset')
@param('adapt.prompt.prompting_target_batch_size')
@param('adapt.prompt.prompting_source_batch_size')
@param('adapt.prompt.backbone_tuning')
@param('adapt.prompt.saliency_tuning')
def prompt(
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
        source_dataset,
        prompting_target_batch_size,
        prompting_source_batch_size,
        ):

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

    from data.prompting import get_unsupervised_prompting_data
    from torch_geometric.loader import DataLoader

    prompting_datasets = get_unsupervised_prompting_data(dataset, source_dataset) # aligning the target dataset to source_dataset (pretrained) dataset    

    prompting_target_loaders = { 'source': DataLoader(prompting_datasets['source'], batch_size=prompting_source_batch_size, shuffle=True),
                                 'target': DataLoader(prompting_datasets['target'], batch_size=prompting_target_batch_size, shuffle=True)}    

    best_acc = 0.
    best_backbone = None
    best_answering = None


    print(("{} frozen gnn | unsupervised tuning prompt | no answering function...".format(epoch)))
    #无监督训练prompt模块
    prompt_model.train()
    model.backbone.eval()
    model.answering.eval()

    prompt_model=prompting_step(prompt_model, opi_pg, model.backbone, prompting_target_loaders['source'], prompting_target_loaders['target'])

    prompt_model.eval()
    model.backbone.eval()

    for e in range(epoch):

        loss_metric.reset()
        acc_metric.reset()
        f1_metric.reset()
        auroc_metric.reset()
        
        print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(e, epoch)))
        model.answering.train()
        answering_with_frozen_prompt_step(loaders['train'], opi_answer, model.backbone, prompt_model, model.answering, e)
                
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
                auroc_metric.update(model(batch), batch.y)
                pbar.set_description(f'Epoch {e} Validation Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
            pbar.close()

        if acc_metric.compute() > best_acc:
            best_acc = acc_metric.compute()
            best_backbone = deepcopy(model.backbone)
            best_answering = deepcopy(model.answering)
    
    model.backbone = best_backbone if best_backbone is not None else model.backbone
    model.answering = best_answering if best_answering is not None else model.answering

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
            auroc_metric.update(model(batch), batch.y)
            pbar.set_description(f'Testing Acc: {acc_metric.compute():.4f}, AUROC: {auroc_metric.compute():.4f}, F1: {f1_metric.compute():.4f}', refresh=True)
        pbar.close()

    return {
        'acc': acc_metric.compute().item(),
        'auroc': auroc_metric.compute().item(),
        'f1': f1_metric.compute().item(),
        'model': model.state_dict(),
    }

@param('adapt.prompt.prompt_epoch')
def prompting_step(PG, opi_pg, gnn, source_dataloader, target_dataloader, prompt_epoch):

    from tqdm import tqdm

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    best_prompting_model = None
    best_prompt_loss = 100000000
    for i in range(1, prompt_epoch + 1):

        target_pbar = tqdm(target_dataloader, total=len(target_dataloader), ncols=100, desc=f'Prompt Epoch {i} Prompting, Loss: batch_loss (average loss)')

        # print('{}-th epoch prompting tuning:\n'.format(i))

        current_epoch_total_loss = 0.
        for target_batch_id, target_grpah_batch in enumerate(target_pbar):  
            target_grpah_batch = target_grpah_batch.to(device)

            prompted_graph = PG(target_grpah_batch)
            target_prompt_graph_embs = gnn(prompted_graph)

            batch_loss = 0.

            for source_batch_id, source_graph_batch in enumerate(source_dataloader):
                
                source_graph_batch = source_graph_batch.to(device)
                source_prompt_graph_embs = gnn(source_graph_batch)
                # print(graph_emb)
                # 如果目标图和源图差异过大，使用对比学习方法可能会导致梯度消失的问题，所以考虑在loss上加入一个极小的bias值，1e-4
                from data.utils import loss_contrastive_learning
                sub_batch_loss = loss_contrastive_learning(target_prompt_graph_embs, source_prompt_graph_embs)
                        
                batch_loss+=sub_batch_loss
           
            batch_loss = batch_loss/len(source_dataloader)
            opi_pg.zero_grad()
            batch_loss.backward()
            opi_pg.step()            
            current_epoch_total_loss += batch_loss.item()
           
            target_pbar.set_description('Prompt Epoch {} Prompting Step, Loss: {:.8f}({:.8f})'.format(i, batch_loss, current_epoch_total_loss/(target_batch_id+1), refresh=True))

        if current_epoch_total_loss < best_prompt_loss:
            best_prompting_model = copy.deepcopy(PG)

        target_pbar.close()
    return best_prompting_model

@param('adapt.prompt.ans_epoch')
def answering_with_frozen_prompt_step(target_loader, opi, gnn, PG, answering, outer_epoch, ans_epoch):

    from tqdm import tqdm

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    for j in range(1, ans_epoch + 1):
        running_loss = 0.
        
        ans_pbar = tqdm(target_loader, total=len(target_loader), ncols=100, desc=f'Outer Epoch {outer_epoch} / Answering Epoch {ans_epoch} Training, Loss: inf')

        for batch_id, train_batch in enumerate(ans_pbar):  # bar2
            # print(train_batch)
            train_batch = train_batch.to(device)
            prompted_graph = PG(train_batch)
            # print(prompted_graph)

            graph_emb = gnn(prompted_graph)

            # print(graph_emb)
            pred = answering(graph_emb)
            # print(pre)
            train_loss = torch.nn.functional.cross_entropy(pred, train_batch.y)

            opi.zero_grad()
            train_loss.backward()
            opi.step()
            running_loss += train_loss.item()

            current_avg_last_loss = running_loss / (batch_id+1)  # loss per batch

            ans_pbar.set_description('Outer Epoch {} / Answer Epoch {} Answering Step | avg loss: {:.8f}'.format(outer_epoch, j, current_avg_last_loss), refresh=True)
            # if batch_id % 5 == 4:  # report every 5 updates
            #     current_avg_last_loss = running_loss / (batch_id+1)  # loss per batch

            #     ans_pbar.set_description('Outer Epoch {} / Answer Epoch {} Answering Step | avg loss: {:.8f}'.format(outer_epoch, j, current_avg_last_loss), refresh=True)
        
        ans_pbar.close()

@param('adapt.epoch')
@param('adapt.prompt.prompt_lr')
@param('adapt.prompt.prompt_weight_decay')
@param('adapt.prompt.ans_lr')
@param('adapt.prompt.ans_weight_decay')
@param('adapt.prompt.backbone_tuning')
@param('adapt.prompt.saliency_tuning')
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

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
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

        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

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