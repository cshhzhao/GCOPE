#!/bin/bash

target_datasets=("cora")

backbone="fagcn"
backbone_tuning=1
split_method="RandomWalk"

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
    echo "storage/reconstruct/${source_dataset_str}_pretrained_model.pt"    

    python src/exec.py --config-file pretrain.json --general.save_dir "storage/reconstruct" --general.reconstruct 0.2 --data.name "${source_dataset_str}" --pretrain.split_method ${split_method}
    
done