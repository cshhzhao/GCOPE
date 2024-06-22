learning_rates=(1e-2 5e-3 1e-3 1e-4 5e-4)
target_datasets=("photo")

backbone="fagcn"
backbone_tuning=1 
split_method="RandomWalk"

few_shot=1
batch_sizes=(100) 


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

    python src/exec.py --config-file pretrain.json --general.save_dir "storage/${backbone}/reconstruct" --general.reconstruct 0.2 --data.name "${source_dataset_str}" --pretrain.split_method ${split_method} --model.backbone.model_type ${backbone}
    
    for lr in "${learning_rates[@]}"
    do
        for bs in "${batch_sizes[@]}"
        do
            python src/exec.py --general.func adapt  --general.save_dir "storage/${backbone}/balanced_few_shot_fine_tune_backbone_with_rec" --general.few_shot $few_shot  --general.reconstruct 0.0 --data.node_feature_dim 100 --data.name $target_dataset --adapt.method finetune --model.backbone.model_type ${backbone} --model.saliency.model_type "none" --adapt.method finetune --adapt.pretrained_file "storage/${backbone}/reconstruct/${source_dataset_str}_pretrained_model.pt" --adapt.finetune.learning_rate $lr --adapt.batch_size $bs --adapt.finetune.backbone_tuning $backbone_tuning        
        done 
    done
done