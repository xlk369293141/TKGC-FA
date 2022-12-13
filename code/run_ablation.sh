#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-01
#SBATCH --output ../tucker_att_ablation.out
#SBATCH --error ../tucker_att_ablation.err

# ICEWS14
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_DFT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_DFT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 0.0 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0

# YAGO15K
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_DFT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_DFT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 0.0 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
