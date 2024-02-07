#!/bin/bash

# 定义 learning_rate 和 batch_size 的数组
learning_rates=(1e-2 1e-3 5e-3 1e-4 5e-4)
# learning_rates=(1e-2)
batch_sizes=(4 8 10 32 64 128)
# target_datasets=("cornell")
# target_datasets=("wisconsin" "texas" "chameleon" "squirrel")
# target_datasets=("wisconsin" "cornell" "texas" "chameleon" "squirrel" "cora" "citeseer" "pubmed" "computers" "photo")
# "cora" "citeseer" "pubmed" "computers" "photo"
target_datasets=("photo")

# 是否调整backbone也就是GNN模型的参数，1为调整，0不调整(只调整head，也就是分类器的参数)
backbone="fagcn"
backbone_tuning=0

few_shot=1
batch_sizes=(100) #few_shot肯定是full batch

seeds=(777)


for seed in "${seeds[@]}"; do
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
        echo $source_dataset_str
        echo $target_dataset
        echo "storage_simgrace/reconstruct/${source_dataset_str}_pretrained_model.pt"
        
        # 确定完pretrain的数据集列表，和prompt的target dataset名称
        # 对每个 learning_rate
        for lr in "${learning_rates[@]}"
        do
            # 对每个 batch_size
            for bs in "${batch_sizes[@]}"
            do
                python src/exec.py --general.func adapt --general.seed ${seed} --general.save_dir "storage_simgrace/balanced_few_shot_ours_with_prog" --general.few_shot $few_shot  --general.reconstruct 0.0 --data.node_feature_dim 100 --data.name $target_dataset --adapt.method prog --data.supervised.ratios 0.1,0.1,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --adapt.pretrained_file "storage_simgrace/reconstruct/${source_dataset_str}_pretrained_model.pt"  --adapt.prompt.ans_lr $lr --adapt.batch_size $bs  --adapt.prompt.backbone_tuning $backbone_tuning --adapt.prompt.source_dataset "${source_dataset_str}" --adapt.batch_size $bs
                # python src/exec.py --general.func adapt --adapt.epoch 20 --general.seed ${seed} --general.save_dir "storage_simgrace/balanced_few_shot_ours_with_prog/prompt_node_5" --general.few_shot $few_shot  --general.reconstruct 0.0 --data.node_feature_dim 100 --data.name $target_dataset --adapt.method prog --data.supervised.ratios 0.1,0.1,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --adapt.pretrained_file "storage_simgrace/reconstruct/${source_dataset_str}_pretrained_model.pt"  --adapt.prompt.ans_lr $lr --adapt.batch_size $bs  --adapt.prompt.backbone_tuning $backbone_tuning --adapt.prompt.source_dataset "${source_dataset_str}" --adapt.batch_size $bs
                # python src/exec.py --general.func adapt --adapt.epoch 20 --general.seed ${seed} --general.save_dir "storage_simgrace/balanced_few_shot_ours_with_prog/prompt_node_10" --general.few_shot $few_shot  --general.reconstruct 0.0 --data.node_feature_dim 100 --data.name $target_dataset --adapt.method prog --data.supervised.ratios 0.1,0.1,0.8 --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --adapt.pretrained_file "storage_simgrace/reconstruct/${source_dataset_str}_pretrained_model.pt"  --adapt.prompt.ans_lr $lr --adapt.batch_size $bs  --adapt.prompt.backbone_tuning $backbone_tuning --adapt.prompt.source_dataset "${source_dataset_str}" --adapt.batch_size $bs
            done 
        done
    done
done