# TKGE-Temp: A typical trial for temporal knowleadge graph completion

## Dependencies
- Python 3.6+
- PyTorch 1.0~1.7
- NumPy 1.17.2+
- tqdm 4.41.1+

## Reproduce the Results

### 1. Preprocess the Datasets
To preprocess the datasets, run the following commands.

```shell script
cd code
python process_datasets.py
```

Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results 
To reproduce the results of CP, ComplEx and RESCAL with
the DURA regularizer on WN18RR, FB15k237 and YAGO3-10,
please run the following commands.

```shell script
#################################### WN18RR ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model CP --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset WN18RR --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 1e-1 --max_epochs 50 \
--valid 5 -train -id 0 -save -weight

# RESCAL
CUDA_VISIBLE_DEVICES=2 python learn.py --dataset WN18RR --model RESCAL --rank 256 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer DURA_RESCAL --reg 1e-1 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

#################################### FB237 ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model CP --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model ComplEx --rank 2000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 100 --regularizer DURA_W --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save

# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset FB237 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 512 --regularizer DURA_RESCAL --reg 5e-2 --max_epochs 200 \
--valid 5 -train -id 0 -save


#################################### YAGO3-10 ####################################
# CP
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model CP --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer DURA_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight

# ComplEx
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model ComplEx --rank 1000 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1000 --regularizer DURA_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save

# RESCAL
CUDA_VISIBLE_DEVICES=0 python learn.py --dataset YAGO3-10 --model RESCAL --rank 512 --optimizer Adagrad \
--learning_rate 1e-1 --batch_size 1024 --regularizer DURA_RESCAL_W --reg 5e-3 --max_epochs 200 \
--valid 5 -train -id 0 -save -weight
```

## Acknowledgement
We refer to the code of [kbc](https://github.com/facebookresearch/kbc) and [DURA](https://github.com/MIRALab-USTC/KGE-DURA). Thanks for their contributions.