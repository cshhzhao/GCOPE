# GCOPE
Implementation of 'All in One and One for All: A Simple yet Effective Method towards Cross-domain Graph Pretraining' accepted for KDD'24.

### Setup Environment

```
1. cd ./environement
2. conda env create -f environment.yml
```

### Run Script

`pretrain.json` is the configuration file of pretraining phase, such as the selected pretraining method, pretraining epoch, etc.

`iterate_supplementary_on_pretrain_with_transferring.sh` is the running file for executing the GCOPE framework, where the shell file gives several options to determine the backbone model (e.g., fagcn, gcn, gat, bwgnn), target downstream transferring dataset (e.g., photo), few_shot (i.e., the num of training samples), backbone_tuning (1 for finetuning the gnn backbone and 0 for freezing the backbone), learning_rates, etc.

Concretely, if you have installed the environment file successfully, you can directly run the following codes to evaluate our proposed GCOPE:

> chmod +x iterate_supplementary_on_pretrain_with_transferring.sh

> ./iterate_supplementary_on_pretrain_with_transferring.sh

If you want to change the backbone or others, just modify the `pretrain.json` or the parameters in `iterate_supplementary_on_pretrain_with_transferring.sh`.