
#!/bin/bash
#SBATCH -p inspur
#SBATCH -w inspur-gpu-01
#SBATCH --output ../tucker_dft_dura_14_2.out
#SBATCH --error ../tucker_dft_dura_14_2.err

CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_DFT2 --rank1 200 --rank2 200 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
--valid 10 -train -id 0 -save -weight --ratio 0.64 --dropout 0.2

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS05-15 --model TuckER_DFT --rank1 200 --rank2 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 512 --regularizer DURA --reg 1e-2 --max_epochs 100 \
# --valid 10 -train -id 0 -save -weight --ratio 0.32

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset GDELT --model TuckER_DE --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 1024 --regularizer NA --reg 1e-1 --max_epochs 100 \
# --valid 20 -train -id 0 -save -weight --ratio 0.32sad

# CUDA_VISIBLE_DEVICES=3 python -u debug.py --dataset GDELT --model TuckER_DE --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer DURA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 0 -save -weight --ratio 0.64

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset WN18RR --model TuckER --rank 200 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer NA --reg 1e-1 --max_epochs 50 \
# --valid 5 -train -id 1 -save

# CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_DFT2 --rank1 400 --rank2 400 --optimizer Adagrad \
# --learning_rate 1e-1 --batch_size 128 --regularizer T_DURA --reg 0.01 --max_epochs 50 \
# --valid 10 -train -id 1 -save -weight --ratio 0.75