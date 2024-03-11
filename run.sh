# FB15kET
CUDA_VISIBLE_DEVICES=1 python3 run.py --model RGCP --dataset FB15kET \
--load_ET --load_KG --neighbor_sampling --neighbor_num 35 --hidden_dim 100 
--lr 0.001 --lr_step 800 --train_batch_size 128 --cuda --num_layers 2 --num_bases 45 --selfloop --lambda_p 0.2 --drop 0.2 --l2 5e-4


# YAGO43kET
CUDA_VISIBLE_DEVICES=1 python3 run.py --model RGCP --dataset YAGO43kET \
--load_ET --load_KG --neighbor_sampling --neighbor_num 35 --hidden_dim 100 \
--lr 0.0001 --lr_step 700 --train_batch_size 16 --cuda --num_layers 2 --num_bases 45 --selfloop --lambda_p 0.2 --drop 0.2 --l2 5e-4