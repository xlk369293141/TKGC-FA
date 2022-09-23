#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-02
#SBATCH --output ../tucker_att_ic14_5.out
#SBATCH --error ../tucker_att_ic14_5.err

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer N3 --reg 1e-2 --reg_t 1e-1 --p 4 --reg_w 1e-3 --max_epochs 50 \
# --valid 5 -train -id 0 -save --ratio 1.0 --dropout 0.1 --no_time_emb --init 0.01

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer N3 --reg 1e-2 --reg_t 0.5 --p 4 --reg_w 1e-3 --max_epochs 50 \
--valid 5 -train -id 1 -save --ratio 0.9 --dropout 0.5 --init 0.001

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_DFT --rank1 200 --rank2 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 256 --regularizer DURA --reg 5e-2 --max_epochs 100 \
# --valid 10 -train -id 0 -save -weight --ratio 0.32

# CUDA_VISIBLE_DEVICES=2 python -u learn.py --dataset GDELT --model TuckER_DFT --rank1 1000 --rank2 1000   --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 256 --regularizer TmpReg --reg 5e-3 --reg_t 0.1 --p 4 --max_epochs 40 \
# --valid 20 -train -id 1 -save -weight --ratio 0.8 --dropout 0.2

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER_con --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer DURA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer ConR --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save