# TKGE-FA: A pytorch implement of TuckER-FA for temporal knowleadge graph completion

## Dependencies
- Python 3.6+
- PyTorch 1.0~1.7
- NumPy 1.17.2+
- tqdm 4.41.1+

## Reproduce the Results

### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
tar -zxvf src_data.tar.gz
cd code
python process_datasets.py
```

Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results 
To reproduce the results of Tucker-FA on ICEWS14, GDELT and YAGO15k,
please run the following commands.

```shell script
#################################### ICEWS14 ####################################
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset ICEWS14 --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-2 --reg_t 1.0 --p 4 --reg_w 1e-2 \
--max_epochs 50 --valid 5 -train -id 10 -save --seed 64504 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0

#################################### GDELT ####################################
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset GDELT --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-1 --reg_t 1.0 --p 4 --reg_w 1e-4 \
--max_epochs 50 --valid 5 -train -id 10 -save --seed 64573 --ratio 0.9 --dropout 0.3 --init 0.01 --mapper 0


#################################### YAGO15k ####################################
CUDA_VISIBLE_DEVICES=0 python -u learn.py --dataset YAGO15k --model TuckER_ATT --rank1 800 --rank2 800 \
--optimizer Adagrad --learning_rate 1e-1 --batch_size 1000 --regularizer L4 --reg 1e-4 --reg_t 0.01 --p 4 --reg_w 1e-3 \
--max_epochs 50 --valid 5 -train -id 10 -save --seed 64670 --ratio 0.75 --dropout 0.3 --init 0.01 --mapper 0
```

## Acknowledgement
We refer to the code of [kbc](https://github.com/facebookresearch/kbc) and [FcaNet](https://github.com/cfzd/FcaNet). Thanks for their contributions.