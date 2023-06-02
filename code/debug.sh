#!/bin/bash
#SBATCH -p inspur
#SBATCH -N 1
#SBATCH --gres=gpu:A100:1
#SBATCH --output ../debug.out
#SBATCH --error ../debug.err


CUDA_VISIBLE_DEVICES=0 python -u more_test.py --dataset YAGO15K --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 25 -test -id 10 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 --no_time_emb \
--checkpoint /home/LAB/xiaolk/TKGC-Temp/logs/TuckER_ATT_L4_YAGO15K_10_64670

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-2 \
--max_epochs 50 --valid 25 -test -id 10  --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
--checkpoint /home/LAB/xiaolk/TKGC-Temp/logs/TuckER_ATT_L4_ICEWS14_10_64504