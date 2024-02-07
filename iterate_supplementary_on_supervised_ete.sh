#!/bin/bash

# 定义 learning_rate 和 batch_size 的数组
learning_rates=(1e-2 1e-3 5e-3 1e-4 5e-4)
# learning_rates=(1e-2)
# batch_sizes=(8 10 32 64 128)
batch_sizes=(8 10 32)
# target_datasets=("cora" "citeseer" "pubmed" "computers" "photo")
# target_datasets=("cornell" "cora" "citeseer" "pubmed" "computers" "photo")
target_datasets=("cora" "citeseer" "pubmed" "wisconsin" "texas" "chameleon" "squirrel" "cornell" "computers" "photo")
# target_datasets=("computers")
# target_datasets=("citeseer")
backbone="fagcn"
few_shot=0
seed=777
# 遍历原始数组
for target_dataset in "${target_datasets[@]}"; do
    # 确定完pretrain的数据集列表，和finetune的target dataset名称
    # 对每个 learning_rate
    for lr in "${learning_rates[@]}"
    do
        # 对每个 batch_size
        for bs in "${batch_sizes[@]}"
        do
            python src/exec.py --general.func ete --general.seed ${seed} --general.save_dir "storage/supervised_ete" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --data.supervised.ratios 0.1,0.1,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --ete.learning_rate $lr --ete.batch_size $bs
        done 
    done    
done