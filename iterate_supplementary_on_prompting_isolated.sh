#!/bin/bash

# debug
# prompt_epochs=(1 3 5 7 10 20 30 40 50 100)
# learning_rates=(1e-3)
# batch_sizes=(4)

# grid_search version 1
# 定义 learning_rate 和 batch_size 的数组
# learning_rates=(1e-3 5e-3 1e-4 5e-4)
# prompt_epochs=(100 10)
# batch_sizes=(4 8 10 32 64 128)

# grid_search version 2 (实际上是1-50,最后一个值(i.e., 100)没有参与实验 prompt_epochs=100的时候参数读不进去，有个bug忽略了)
prompt_epochs=(1 3 5 7 10 50 100)
learning_rates=(1e-3 5e-3 1e-4 5e-4)
batch_sizes=(4 8 10 32 64 128)

# target_datasets=("cora" "citeseer" "pubmed" "computers" "photo")
# target_datasets=("cornell" "texas" "wisconsin" "chameleon" "squirrel")
# target_datasets=("cora" "citeseer" "photo" "pubmed" "computers")
# target_datasets=("texas" "wisconsin")
target_datasets=("computers")

# 是否调整backbone也就是GNN模型的参数，1为调整，0不调整(只调整head，也就是分类器的参数)
backbone_tuning=0

# 要移除的元素
# remove="cora"
# 新数组，用于存储差集结果

# 遍历原始数组
for target_dataset in "${target_datasets[@]}"; do
    source_dataset_str=""
    datasets=("wisconsin" "texas" "cornell" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "computers" "photo")
    for dataset in "${datasets[@]}"; do
        # 如果当前元素不是要移除的元素，则添加到新数组中
        if [ "$dataset" != "$target_dataset" ]; then
            source_dataset_str+="${dataset},"
        fi
    done

    # 输入的SubsetOf = ['cora', 'citeseer']其实只要是list对象就行了，也就是 字符串"cora,citeseer"就可以了
    source_dataset_str="${source_dataset_str%,}"
    
    # 确定完pretrain的数据集列表，和prompt的target dataset名称
    # 对每个 learning_rate
    for lr in "${learning_rates[@]}"
    do
        for bs in "${batch_sizes[@]}"
        do
            # 对每个 batch_size
            for pe in "${prompt_epochs[@]}$"
            do
                python src/exec.py --general.func adapt --adapt.epoch 10 --data.node_feature_dim 100 --data.clustered.num_parts 100 --data.name $target_dataset --data.supervised.ratios 0.1,0.1,0.8 --model.saliency.model_type "none" --adapt.method prompt --adapt.pretrained_file "storage/tmp/${source_dataset_str}_pretrained_model.pt" --general.save_dir "storage/prompt" --adapt.prompt.ans_lr $lr --adapt.batch_size $bs --adapt.prompt.prompt_epoch $pe --adapt.prompt.ans_epoch 1 --adapt.prompt.backbone_tuning $backbone_tuning --adapt.prompt.source_dataset "${source_dataset_str}"
            done 
        done    
    done
done
# 记得调整ans_epoch和prompt_epoch的大小 还有--data.clustered.num_parts 500 for heterophily;100 for homophily