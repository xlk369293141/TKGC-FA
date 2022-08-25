#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH --output ../tucker_dft_debug3.out
#SBATCH --error ../tucker_dft_debug3.err

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_DFT --rank1 200 --rank2 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 0.1 --reg_t 1.5 --p 4 --max_epochs 50 \
--valid 10 -train -id 1 -save -weight --ratio 0.8 --dropout 0.2

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_DFT --rank1 200 --rank2 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 512 --regularizer DURA --reg 1e-2 --max_epochs 100 \
# --valid 10 -train -id 0 -save -weight --ratio 0.32

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset GDELT --model TuckER_DE --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 1024 --regularizer NA --reg 1e-1 --max_epochs 100 \
# --valid 20 -train -id 0 -save -weight --ratio 0.32sad

# CUDA_VISIBLE_DEVICES=3 python -u debug.py --dataset GDELT --model TuckER_DE --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer DURA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 0 -save -weight --ratio 0.64

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save