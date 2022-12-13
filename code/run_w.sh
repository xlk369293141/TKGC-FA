#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-03
#SBATCH --output ../tucker_att_debug_w.out
#SBATCH --error ../tucker_att_debug_w.err

i=0 

for reg_name in 0 1 2
do
    for reg_w in 0.00001 0.0001 0.001 0.01 0.1
    do
        ((i++))
        echo "RUN $i:----------"
        echo "dropout: $reg_name   ratio: $reg_w"
        CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 400 --rank2 400 \
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --core_type $reg_name --reg_w $reg_w \
        --max_epochs 50 --valid 5 -train -id 0 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
        echo "----------------"
    done
done

for reg_name in 0 1 2
do
    for reg_w in 0.0001 0.001 0.01 0.1 1.0
    do
        ((i++))
        echo "RUN $i:----------"
        echo "dropout: $reg_name   ratio: $reg_w"
        CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15K --model TuckER_ATT --rank1 400 --rank2 400 \
        --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 1e-2 --p 4 --core_type $reg_name --reg_w $reg_w \
        --max_epochs 50 --valid 5 -train -id 0 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
        echo "----------------"
    done
done

# for reg_name in CoreReg, CoreRegNew, CoreRegOld
# do
#     for reg_w in 0.0 0.0001 0.001 0.001 0.1 1.0
#     do
#         ((i++))
#         echo "RUN $i:----------"
#         echo "dropout: $dropout   ratio: $ratio"
#         CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_ATT --rank1 400 --rank2 400 \
#         --optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-3 \
#         --max_epochs 50 --valid 10 -train -id 0 -save --seed 64064 --ratio $ratio --dropout $dropout --init 0.01 --mapper 0
#         echo "----------------"
#     done
# done