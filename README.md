# R-GCP_dgl
This repository provides the source code of **Relational-Graph Convolutional PonderNet (R-GCP)** for Entity Classification in Knowledge Graphs on FB15kET and YAGO43kET datasets. The model R-GCP is implemented using **[DGL](https://github.com/dmlc/dgl)** library.

## Setup
To run this repository, you must install the following dependencies first:
* Compatible with PyTorch 1.13.1 and Python 3.8.17 and DGL 0.6.1.
* Dependencies can be installed using `requirements.txt`.

## Training
You can learn the representations on datasets **FB15kET** and **YAGO43kET** through the following commands. (Run `run.sh`)
```shell
# FB15kET
CUDA_VISIBLE_DEVICES=1 python3 run.py --model RGCP --dataset FB15kET \
--load_ET --load_KG --neighbor_sampling --neighbor_num 35 --hidden_dim 100 
--lr 0.001 --lr_step 800 --train_batch_size 128 --cuda --num_layers 2 --num_bases 45 --selfloop --lambda_p 0.2 --drop 0.2 --l2 5e-4

# YAGO43kET
CUDA_VISIBLE_DEVICES=1 python3 run.py --model RGCP --dataset YAGO43kET \
--load_ET --load_KG --neighbor_sampling --neighbor_num 35 --hidden_dim 100 \
--lr 0.0001 --lr_step 700 --train_batch_size 16 --cuda --num_layers 2 --num_bases 45 --selfloop --lambda_p 0.2 --drop 0.2 --l2 5e-4
```
### Note
- Results depend on random seed and will vary between re-runs.
- You can obtain the Confidence Intervals(CI) of results and visulize them by running following command.
```
time python CI_Calc.py
```
- The pre-trained best models are avaliable [here](https://drive.google.com/drive/folders/1zDGQv1gtDUq8ichbs_rE0VvjN8tWhaNV?usp=drive_link). You can download them to this repository and run eval.sh directly.

## Baselines
- Baselines are available in the folder /Baseline, while R-GCN and CompGCN are adapted from this [repo](https://github.com/CCIIPLab/CET).

## Acknowledgement
We refer to the code of **[CET](https://github.com/CCIIPLab/CET)**. Thanks for their contributions.
