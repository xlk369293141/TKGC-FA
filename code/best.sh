#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH --output ../tucker_att_best.out
#SBATCH --error ../tucker_att_best.err


# ICEWS05-15
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-5 \
--max_epochs 50 --valid 50 -train -id 10 -save --seed 27650 --ratio 0.9 --dropout 0.3 --init 0.01 --mapper 0

# ICEWS14
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 25 -train -id 10 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0

# YAGO15K
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 25 -train -id 10 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 --no_time_emb


# 9643