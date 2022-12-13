#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-01
#SBATCH --output ../tucker_att_yago_mapper_2.out
#SBATCH --error ../tucker_att_yago_mapper_2.err

i=0

# for mapper in 0 1 2 3 4 5 6 7 8 9 10 11 12
# do
#     i=$[$i+1]
#     echo "RUN $i:----------"
#     CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
#     --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
#     --max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper $mapper --no_time_emb
#     echo "----------------"
# done


i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 5 8 10 --no_time_emb
echo "----------------"

i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 2 3 5 8 10 --no_time_emb
echo "----------------"

i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 2 3 5 6 7 8 9 10 12 --no_time_emb
echo "----------------"

i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 8 5 10 2 3 0 12 9 6 7 4 11  --no_time_emb
echo "----------------"