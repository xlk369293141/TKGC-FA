#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-04
#SBATCH --output ../tucker_att_ic15.out
#SBATCH --error ../tucker_att_ic15.err

i=0
# echo "RUN $i:----------"
# # CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 400 --rank2 400 \
# #             --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 0.0 --reg_t 0.0 --core CoreReg --reg_w 0.0 \
# #             --max_epochs 50 --valid 10 -train -id 1 -save --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
# echo "----------------"

for reg_t in 0.5 0.2 1.0
do
    for reg in 0.5 0.005 0.01
    do
        for reg_w in 0.0001 0.00001 0.000001
        do
            i=$[$i+1]
            echo "RUN $i:----------"
            CUDA_VISIBLE_DEVICES=2 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 800 --rank2 800 \
            --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg $reg --reg_t $reg_t --p 4 --reg_w $reg_w \
            --max_epochs 50 --valid 50 -train -id 1 -save --ratio 0.9 --dropout 0.3 --init 0.01 --mapper 0
            echo "----------------"
            done
    done
done