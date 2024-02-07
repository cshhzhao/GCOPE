from importlib import import_module
from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import OneOf, File, ListOfFloats, Folder, SubsetOf, BoolAsInt
import argparse
import os

Section('general', 'General Configs').params(
    func = Param(OneOf(['pretrain', 'adapt', 'ete']), required=True),
    cache_dir = Param(Folder(True), default='storage/.cache'),
    save_dir = Param(Folder(True), required=True, default=f'storage/tmp'),
    seed = Param(int, default=777, desc='seed for general randomness'),
    few_shot = Param(int, default=1, desc='few-shot node classification'),
    reconstruct = Param(float, default=0.0),
)

Section('model.backbone', 'Backbone General Configs').params(
    model_type = Param(OneOf(['fagcn', 'bwgnn','gcn', 'gat']), default='fagcn', desc='backbone model to use'),
    hid_dim = Param(int, default=128),
)

Section('model.backbone.fagcn', 'FAGCN Model Configs').enable_if(
    lambda cfg: cfg['model.backbone.model_type'] == 'fagcn'
).params(
    num_conv_layers = Param(int, default=2),
    dropout = Param(float, default=0.2),
    epsilon = Param(float, default=0.1),
)

Section('model.backbone.gcn', 'FAGCN Model Configs').enable_if(
    lambda cfg: cfg['model.backbone.model_type'] == 'gcn'
).params(
    num_conv_layers = Param(int, default=2),
    dropout = Param(float, default=0.2),
)

Section('model.backbone.gat', 'FAGCN Model Configs').enable_if(
    lambda cfg: cfg['model.backbone.model_type'] == 'gat'
).params(
    num_conv_layers = Param(int, default=2),
    dropout = Param(float, default=0.2),
    head = Param(int, default=8),
)

Section('model.saliency', 'Saliency Model Configs').params(
    model_type = Param(OneOf(['mlp', 'transformer', 'none']), default='transformer', desc='saliency model to use'),
)

Section('model.saliency.mlp').enable_if(
    lambda cfg: cfg['model.saliency.model_type'] == 'mlp'
).params(
    hid_dim = Param(int, default=4096),
    num_layers = Param(int, default=2),
)

Section('model.saliency.transformer').enable_if(
    lambda cfg: cfg['model.saliency.model_type'] == 'transformer'
).params(
    token_hid_dim = Param(int, default=4096),
    num_blocks = Param(int, default=2),
)

Section('model.answering', 'Answering General Configs').enable_if(
    lambda cfg: cfg['general.func'] in ['adapt', 'ete']
).params(
    model_type = Param(OneOf(['mlp']), default='mlp'),
)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      

Section('model.answering.mlp').enable_if(
    lambda cfg: cfg['model.answering.model_type'] == 'mlp'
).params(
    num_layers = Param(int, default=2),
)

Section('data', 'Data Configs').params(
    name = Param(SubsetOf([
        'wisconsin', 'texas', 'cornell', 'chameleon', 'squirrel',
        'cora', 'citeseer', 'pubmed', 'computers', 'photo',
        ]), required=True),
    seed = Param(int, default=777, desc='seed for train/val/test split, fix this to get the same dataset'),
    node_feature_dim = Param(int, default=0, desc='0: use only structural information, >0: use node features, SVD if lower than the actual number of features else Padding'),
)

Section('data.clustered', 'Clustered Data Configs').enable_if(
    lambda cfg: cfg['general.func'] in ['pretrain', 'prompt']
).params(
    num_parts = Param(int, default=0),
)

Section('data.supervised', 'Supervised Data Configs').enable_if(
    lambda cfg: cfg['general.func'] in ['adapt', 'ete']
).params(
    ratios = Param(ListOfFloats(), required=True, desc='train/tal/test split ratios'),
)

Section('pretrain', 'Pretraining Configs').enable_if(
    lambda cfg: cfg['general.func'] == 'pretrain'
).params(
    method = Param(OneOf(['graphcl', 'simgrace']), default='graphcl'),
    learning_rate = Param(float, default=1e-2),
    weight_decay = Param(float, default=1e-5),
    epoch = Param(int, default=100),
    batch_size = Param(int, default=10),
    noise_switch = Param(BoolAsInt(), default=False), #shell文件的输入是1或0
    # cross_link = Param(BoolAsInt(), default=False),
    cross_link = Param(int, default=0),
    cross_link_ablation = Param(BoolAsInt(), default=False), # 用于cross_link的Ablation Study
    dynamic_edge = Param(OneOf(['internal', 'external', 'internal_external', 'similarity', 'none']), default='none'), #用于控制virtual node的边和datasets相连接是可学习的还是固定的。
    dynamic_prune = Param(float, default=0.1),
    cl_init_method = Param(OneOf(['mean', 'sum', 'learnable', 'simple', 'none']), default='learnable'),
    split_method = Param(OneOf(['metis', 'RandomWalk']), default='RandomWalk'),
)
# internal指的是虚拟节点之和对应的单个dataset的点做相似性计算，连边，与其他的虚点和数据集不做操作。
# external指的是虚拟节点和对应的dataset全连接，和其他的虚点以及数据集上的点是根据similarity的动态连边
# internal_external指的是每个数据集都可能和任意数据集和任意虚点根据similarity动态相连

Section('adapt', 'Adaptation Configs').enable_if(
    lambda cfg: cfg['general.func'] == 'adapt'
).params(
    repeat_times = Param(int, default=5),
    method = Param(OneOf(['finetune', 'prompt','prog']), default='finetune'),
    pretrained_file = Param(File(), required=True,default='storage/tmp/pretrained_model.pt'),
    epoch = Param(int, default=100),
    batch_size = Param(int, default=10),
)

Section('adapt.prompt', 'Prompt Configs').enable_if(
    lambda cfg: cfg['adapt.method'] in ['prompt', 'prog']
).params(
    prompt_lr = Param(float, default=1e-4),
    # prompt_weight_decay = Param(float, default=1e-5),
    # ans_lr = Param(float, default=1e-4),
    # ans_weight_decay = Param(float, default=1e-5),
    # prompt_lr = Param(float, default=0.002),
    prompt_weight_decay = Param(float, default=1e-5),
    prompt_epoch = Param(int, default = 1),
    ans_lr = Param(float, default=1e-2),    
    ans_weight_decay = Param(float, default=1e-5),
    ans_epoch = Param(int, default = 1),
    backbone_tuning = Param(BoolAsInt(), default=False),
    saliency_tuning = Param(BoolAsInt(), default=False),    
    cross_prune = Param(float, default = 0.3),
    inner_prune = Param(float, default = 0.1),
    edge_attr_dim = Param(int, default = 0),
    prompting_target_batch_size = Param(int, default=128),
    prompting_source_batch_size = Param(int, default=2048),
    # prompting_target_batch_size = Param(int, default=10),
    # prompting_source_batch_size = Param(int, default=64),    
    cross_link = Param(BoolAsInt(), default=True),
    source_dataset = Param(SubsetOf(['wisconsin', 'texas', 'cornell', 'chameleon', 'squirrel','cora', 'citeseer', 'pubmed', 'computers', 'photo',])),
)

Section('adapt.finetune', 'Finetune Configs').enable_if(
    lambda cfg: cfg['adapt.method'] == 'finetune'
).params(
    backbone_tuning = Param(BoolAsInt(), default=False),
    saliency_tuning = Param(BoolAsInt(), default=False),
    learning_rate = Param(float, default=1e-4),
    weight_decay = Param(float, default=1e-5),
)

Section('ete', 'End-to-End Training Configs').enable_if(
    lambda cfg: cfg['general.func'] == 'ete'
).params(
    epoch = Param(int, default=100),
    batch_size = Param(int, default=10),
    learning_rate = Param(float, default=1e-4),
    weight_decay = Param(float, default=1e-5),
    repeat_times = Param(int, default=5),
)


@param('general.func')
@param('general.seed')
@param('data.name')
def run(func, seed,name):
    import sys
    sys.path.append('src')
    # Fix all randomness
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    import_module(f'functional.{func}').run()


if __name__ == '__main__':
    config = get_current_config()
    parser = argparse.ArgumentParser("All in One: Union of Homophily and Heterophily Graphs")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate()
    config.get_all_config(dump_path=os.path.join(config['general.save_dir'], 'config.json'))
    run()