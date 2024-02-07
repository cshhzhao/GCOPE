#!/bin/bash

# 定义 learning_rate 和 batch_size 的数组
learning_rates=(1e-2 1e-3 5e-3 1e-4 5e-4)
batch_sizes=(4 8 10 32 64 128)

# target_datasets=("wisconsin" "texas" "chameleon" "squirrel")
# target_datasets=("pubmed" "cora" "citeseer" "computers" "photo" "wisconsin" "cornell" "texas" "chameleon" "squirrel")
# target_datasets=("pubmed" "cora" "citeseer" "computers" "photo" "wisconsin" "cornell" "texas")
# target_datasets=("cora" "citeseer" "pubmed" "computers" "photo")
# target_datasets=("pubmed" "cora" "citeseer" "computers" "photo")
target_datasets=("wisconsin" "texas" "cornell" "chameleon" "squirrel")
# target_datasets=("citeseer")
source_dataset_str="pubmed"

# 要移除的元素
# remove="cora"
# 新数组，用于存储差集结果
backbone="fagcn"
backbone_tuning=1
split_method="RandomWalk"

few_shot=1
batch_sizes=(100) #few_shot肯定是full batch

# 遍历原始数组
for target_dataset in "${target_datasets[@]}"; do

    echo $target_dataset
    echo "storage/single_pretrains/${source_dataset_str}_pretrained_model.pt"    

    # 执行 exec.py 文件并传递参数
    python src/exec.py --config-file pretrain_single.json --general.save_dir "storage/single_pretrain" --data.name "${source_dataset_str}" --pretrain.split_method ${split_method}
    
    # 确定完pretrain的数据集列表，和finetune的target dataset名称
    # 对每个 learning_rate
    for lr in "${learning_rates[@]}"
    do
        # 对每个 batch_size
        for bs in "${batch_sizes[@]}"
        do
            python src/exec.py --general.func adapt  --general.save_dir "storage/balanced_few_shot_fine_tune_backbone_with_single_pretrain/pubmed" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --adapt.method finetune --data.supervised.ratios 0.1,0.1,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --adapt.method finetune --adapt.pretrained_file "storage/single_pretrain/${source_dataset_str}_pretrained_model.pt" --adapt.finetune.learning_rate $lr --adapt.batch_size $bs --adapt.finetune.backbone_tuning $backbone_tuning
        done 
    done
done