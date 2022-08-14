#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-03
#SBATCH --output ../complex_de_dura_ic14.out
#SBATCH --error ../complex_de_dura_ic14.err

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_DE --rank 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer DURA --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 1 -save -weight --ratio 0.64

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model ComplEx_DE --rank 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 500 --regularizer DURA --reg 1e-1 --max_epochs 100 \
--valid 10 -train -id 0 -save -weight --ratio 0.32

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset GDELT --model TuckER_DE --rank 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer NA --reg 1e-1 --max_epochs 100 \
--valid 20 -train -id 0 -save -weight --ratio 0.32






# CUDA_VISIBLE_DEVICES=3 python -u debug.py --dataset GDELT --model TuckER_DE --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer DURA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 0 -save -weight --ratio 0.64

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save