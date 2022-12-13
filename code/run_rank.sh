#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-03
#SBATCH --output ../tucker_att_rank.out
#SBATCH --error ../tucker_att_rank.err

# YAGO15K
for rank in 32 64 100 200 400
do
    CUDA_VISIBLE_DEVICES=2 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 $rank  --rank2 $rank  \
    --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
    --max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 --no_time_emb
done

# # ICEWS14
# for rank in 32 64 100 200 400
# do
#     CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 $rank --rank2 $rank  \
#     --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
#     --max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
# done