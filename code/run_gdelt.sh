#!/bin/bash
#SBATCH -p inspur
#SBATCH -N 1
#SBATCH --gres=gpu:A100:1
#SBATCH --output ../tucker_att_gdelt_1.out
#SBATCH --error ../tucker_att_gdelt_1.err

# i=0
# for reg_t in 1.0
# do
#     for reg in 0.1
#     do
#         for reg_w in 0.001 0.0001 0.00001
#         do
#             i=$[$i+1]
#             echo "RUN $i:----------"
#             echo "Reg: $reg,  Reg_t:$reg_t, Reg_w:$reg_w"
#             CUDA_VISIBLE_DEVICES=3 python -u learn.py --dataset GDELT --model TuckER_ATT --rank1 800 --rank2 800 \
#             --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg $reg --reg_t $reg_t --p 4 --reg_w $reg_w \
#             --max_epochs 50 --valid 100 -train -id 2 -save --ratio 0.9 --dropout 0.3 --init 0.01 --mapper 0
#             echo "----------------"
#             done
#     done
# done

#GDELT
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset GDELT --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-1 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 45 --max_epochs 50 --valid 50 -train -id 10 -save --seed 64573 --ratio 0.9 --dropout 0.3 --init 0.01 --mapper 0