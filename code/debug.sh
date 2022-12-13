#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH --output ../debug.out
#SBATCH --error ../debug.err


CUDA_VISIBLE_DEVICES=1 python -u debug.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 0 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 --no_time_emb \
--checkpoint /home/LAB/xiaolk/TKGC-New/TKGC-Temp/logs/TuckER_ATT_L4_YAGO15K_0_64670