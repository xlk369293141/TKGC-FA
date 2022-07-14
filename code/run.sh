#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-05
#SBATCH --output yago_complex_dura.out
#SBATCH --error yago_complex_dura.err

# CUDA_VISIBLE_DEVICES=0 python -u -m learn.py --dataset WN18RR --model ComplEx --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 0 -save -weight

# CUDA_VISIBLE_DEVICES=2 python -u -m learn.py --dataset FB237 --model ComplEx --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 5e-2 --max_epochs 200 \
# --valid 5 -train -id 0 -save

CUDA_VISIBLE_DEVICES=3 python -u -m learn.py --dataset YAGO3-10 --model ComplEx --rank 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer DURA_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save