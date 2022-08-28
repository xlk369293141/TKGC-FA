#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-05
#SBATCH --output ../complex_dft_debug.out
#SBATCH --error ../complex_dft_debug.err

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model ComplEx_DFT --rank1 200 --rank2 200   --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --reg_t 0.8 --p 4 --max_epochs 50 \
--valid 10 -train -id 0 -save -weight --ratio 0.6 --dropout 0.1

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_DFT --rank1 200 --rank2 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 256 --regularizer DURA --reg 5e-2 --max_epochs 100 \
# --valid 10 -train -id 0 -save -weight --ratio 0.32

# CUDA_VISIBLE_DEVICES=2 python -u learn.py --dataset GDELT --model TuckER_DFT --rank1 1000 --rank2 1000   --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 256 --regularizer TmpReg --reg 5e-3 --reg_t 0.1 --p 4 --max_epochs 50 \
# --valid 10 -train -id 1 -save -weight --ratio 0.8 --dropout 0.2

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer DURA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer ConR --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save