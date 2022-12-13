#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-03
#SBATCH --output ../tucker_att_ic15_mapper.out
#SBATCH --error ../tucker_att_ic15_mapper.err

i=0

for mapper in 0 1 2 3 4 5 6 7 8 9 10 11 12
do
    i=$[$i+1]
    echo "RUN $i:----------"
    CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 400 --rank2 400 \
    --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-4 \
    --max_epochs 50 --valid 50 -train -id 0 -save --ratio 0.9 --dropout 0.3 --init 0.01 --mapper $mapper
    echo "----------------"

    i=$[$i+1]
    echo "RUN $i:----------"
    CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 400 --rank2 400 \
    --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-5 \
    --max_epochs 50 --valid 50 -train -id 0 -save --ratio 0.9 --dropout 0.3 --init 0.01 --mapper $mapper
    echo "----------------"
done