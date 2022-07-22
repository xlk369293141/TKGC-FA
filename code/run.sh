#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-01
#SBATCH --output tucker_de.out
#SBATCH --error tucker_de.err

# CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save

CUDA_VISIBLE_DEVICES=1 python learn.py --dataset ICEWS14 --model TuckER_DE --rank 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 0 -save -weight