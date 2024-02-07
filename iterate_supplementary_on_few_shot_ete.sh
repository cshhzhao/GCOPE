#!/bin/bash

# 定义 learning_rate 和 batch_size 的数组
# learning_rates=(1e-2 1e-3 5e-3 1e-4 5e-4)
# learning_rates=(1e-2)
learning_rates=(1e-3 5e-3 1e-4 5e-4)
batch_sizes=(8 10 32 64 128)
# target_datasets=("cora" "citeseer" "pubmed" "computers" "photo")
# target_datasets=("cornell" "cora" "citeseer" "pubmed" "computers" "photo")
target_datasets=("wisconsin" "cornell" "texas" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "computers" "photo")
# target_datasets=("computers")

# backbone="gcn"
backbone="gat"
few_shot=3
batch_sizes=(100) #few_shot肯定是full batch
seeds=(777)
# seeds=(333 444 555 666 777 888 999)

# 遍历原始数组
for seed in "${seeds[@]}"; do
    for target_dataset in "${target_datasets[@]}"; do

        # 确定完pretrain的数据集列表，和finetune的target dataset名称
        # 对每个 learning_rate
        for lr in "${learning_rates[@]}"
        do
            # 对每个 batch_size
            for bs in "${batch_sizes[@]}"
            do
                # python src/exec.py --general.func ete --general.seed ${seed} --general.save_dir "storage/balanced_few_shot_ete/5_shot" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --data.supervised.ratios 0.01,0.19,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --ete.learning_rate $lr --ete.batch_size $bs
                # python src/exec.py --general.func ete --general.seed ${seed} --general.save_dir "storage/balanced_few_shot_ete_gcn" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --data.supervised.ratios 0.01,0.19,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --ete.learning_rate $lr --ete.batch_size $bs
                # python src/exec.py --general.func ete --general.seed ${seed} --general.save_dir "storage/balanced_few_shot_ete_gcn/5_shot" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --data.supervised.ratios 0.01,0.19,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --ete.learning_rate $lr --ete.batch_size $bs
                python src/exec.py --general.func ete --general.seed ${seed} --general.save_dir "storage/balanced_few_shot_ete_gat/3_shot" --general.few_shot $few_shot  --data.node_feature_dim 100 --data.name $target_dataset --data.supervised.ratios 0.01,0.19,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --ete.learning_rate $lr --ete.batch_size $bs
            done 
        done    
    done
done