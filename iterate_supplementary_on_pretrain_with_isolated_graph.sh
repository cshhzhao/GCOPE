#!/bin/bash

# 定义 learning_rate 和 batch_size 的数组
learning_rates=(1e-2 1e-3 5e-3 1e-4 5e-4)

target_datasets=("cora")


backbone="fagcn"
backbone_tuning=1
cross_link=0
split_method="RandomWalk"

few_shot=1
batch_sizes=(100) #few_shot肯定是full batch

# 遍历原始数组
for target_dataset in "${target_datasets[@]}"; do
    source_dataset_str=""
    datasets=("wisconsin" "texas" "cornell" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "computers" "photo")
    for dataset in "${datasets[@]}"; do
        if [ "$dataset" != "$target_dataset" ]; then
            source_dataset_str+="${dataset},"
        fi
    done

    source_dataset_str="${source_dataset_str%,}"
    echo $source_dataset_str
    echo $target_dataset
    echo "storage/isolated_pretrain/${source_dataset_str}_pretrained_model.pt"

    python src/exec.py --config-file pretrain_isolated.json --general.save_dir "storage/isolated_pretrain" --general.reconstruct 0.0 --pretrain.cross_link ${cross_link} --data.name "${source_dataset_str}" --pretrain.split_method ${split_method}
    
    for lr in "${learning_rates[@]}"
    do
        # 对每个 batch_size
        for bs in "${batch_sizes[@]}"
        do
            python src/exec.py --general.func adapt  --general.save_dir "storage/balanced_few_shot_fine_tune_backbone_with_isolated" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --adapt.method finetune --data.supervised.ratios 0.1,0.1,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --adapt.method finetune --adapt.pretrained_file "storage/isolated_pretrain/${source_dataset_str}_pretrained_model.pt" --adapt.finetune.learning_rate $lr --adapt.batch_size $bs --adapt.finetune.backbone_tuning $backbone_tuning
        done 
    done
done