#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH --output ../tucker_att_ic15_debug.out
#SBATCH --error ../tucker_att_ic_15_debug.err

i=0 

# for dropout in 0.5 0.4 0.3
# do
#     for ratio in 0.9 0.75 0.6
#     do
#         ((i++))
#         echo "RUN $i:----------"
#         echo "dropout: $dropout   ratio: $ratio"
#         CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
#         --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
#         --max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio $ratio --dropout $dropout --init 0.01 --mapper 0
#         echo "----------------"
#     done
# done

# for dropout in 0.5 0.4 0.3
# do
#     for ratio in 0.9 0.75 0.6
#     do
#         ((i++))
#         echo "RUN $i:----------"
#         echo "dropout: $dropout   ratio: $ratio"
#         CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
#         --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
#         --max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio $ratio --dropout $dropout --init 0.01 --mapper 0
#         echo "----------------"
#     done
# done

for dropout in 0.5 0.4 0.3
do
    for ratio in 0.9 0.75 0.6
    do
        ((i++))
        echo "RUN $i:----------"
        echo "dropout: $dropout   ratio: $ratio"
        CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 400 --rank2 400 \
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-3 \
        --max_epochs 50 --valid 50 -train -id 0 -save --ratio $ratio --dropout $dropout --init 0.01 --mapper 0
        echo "----------------"
    done
done