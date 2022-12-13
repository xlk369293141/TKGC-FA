#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-01
#SBATCH --output ../tucker_att_ic14_mapper_2.out
#SBATCH --error ../tucker_att_ic14_mapper_2.err

i=0

# for mapper in 0 1 2 3 4 5 6 7 8 9 10 11 12
# do
#     i=$[$i+1]
#     echo "RUN $i:----------"
#     CUDA_VISIBLE_DEVICES=1 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
#     --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
#     --max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper $mapper
#     echo "----------------"
# done




i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=1 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 5 6
echo "----------------"

i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=1 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 5 6 9 10 12
echo "----------------"

i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=1 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 1 4 5 6 8 9 10 11 12
echo "----------------"

i=$[$i+1]
echo "RUN $i:----------"
CUDA_VISIBLE_DEVICES=1 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0 1 2 4 5 6 7 8 9 10 11 12
echo "----------------"