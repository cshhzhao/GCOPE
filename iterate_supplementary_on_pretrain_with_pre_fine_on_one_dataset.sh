#!/bin/bash

# 定义 learning_rate 和 batch_size 的数组
learning_rates=(1e-2 1e-3 5e-3 1e-4 5e-4)
batch_sizes=(4 8 10 32 64 128)
# target_datasets=("wisconsin" "cornell" "texas" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "computers" "photo")
target_datasets=("cornell")

backbone_tuning=1
cross_link=0

few_shot=1
batch_sizes=(100) #few_shot肯定是full batch

# 遍历原始数组
for target_dataset in "${target_datasets[@]}"; do

    echo $target_dataset
    echo "storage/tmp/${target_datasets}_pretrained_model.pt"

    # 执行 exec.py 文件并传递参数
    if [ "cornell" == "$target_dataset" ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 10
    elif [ "wisconsin" == "$target_dataset" ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 10
    elif [ "texas" == "$target_dataset" ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 10
    elif [ "chameleon" == "$target_dataset" ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 10
    elif [ "squirrel" == "$target_dataset" ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 10
    elif [ "cora" == "$target_dataset"  ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 150 #注意这个150，仅限于learnable方法
    elif [ "computers" == "$target_dataset"  ]; then
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 140 #注意这个150，仅限于learnable方法
    else
        python src/exec.py --config-file pretrain.json --pretrain.cross_link ${cross_link} --data.name "${target_datasets}" --data.clustered.num_parts 100
    fi    
    
    # 确定完pretrain的数据集列表，和finetune的target dataset名称
    # 对每个 learning_rate
    for lr in "${learning_rates[@]}"
    do
        # 对每个 batch_size
        for bs in "${batch_sizes[@]}"
        do
            python src/exec.py --general.func adapt  --general.save_dir "storage/balanced_few_shot_pre_fine_on_one_dataset" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --adapt.method finetune --data.supervised.ratios 0.1,0.1,0.8 --model.saliency.model_type "none" --adapt.method finetune --adapt.pretrained_file "storage/tmp/${target_datasets}_pretrained_model.pt" --adapt.finetune.learning_rate $lr --adapt.batch_size $bs --adapt.finetune.backbone_tuning $backbone_tuning
        done 
    done
done