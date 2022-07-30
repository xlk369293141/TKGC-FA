#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-01
#SBATCH --output ../complex_de.out
#SBATCH --error ../complex_de.err

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save

CUDA_VISIBLE_DEVICES=1 python -u learn.py --dataset ICEWS14 --model ComplEx_DE --rank 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 0 -save -weight --ratio 0.64

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset FB237 --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 0 -save -weight --ratio 0.64