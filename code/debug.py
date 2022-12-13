import os
import json
import argparse
import numpy as np

import torch
from torch import optim

from datasets import Dataset
from models import *
from regularizers import *
from optimizers import KBCOptimizer

datasets = ['WN18RR', 'FB237', 'YAGO3-10', 'ICEWS14', 'ICEWS05-15', 'GDELT', 'YAGO15K']

parser = argparse.ArgumentParser(
    description="Tensor Factorization for Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--model', type=str, default='CP'
)

parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=10, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank1', default=200, type=int,
    help="Factorization rank for entity."
)
parser.add_argument(
    '--rank2', default=200, type=int,
    help="Factorization rank for relation or timestamp."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--reg_t', default=0, type=float,
    help="Time Regularization weight"
)
parser.add_argument(
    '--reg_w', default=0, type=float,
    help="Time Regularization weight"
)
parser.add_argument(
    '--p', default=2, type=int,
    help="p-norm of Time Regularization"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument(
    '--ratio', default=0.64, type=float,
    help="the ratio of temporal embedding dimension"
)

parser.add_argument(
    '--dropout', default=0.1, type=float,
    help="dropout rate for fact network"
)

parser.add_argument(
    '--num_heads', default=1, type=int,
    help="The number of frequencies chosen"
)

parser.add_argument(
    '--num_tiles', default=1, type=int,
    help="The number of tiles on the time sequence"
)

parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)

parser.add_argument(
    '--seed', default=-1, type=int, 
    help="Use a specific embedding for non temporal relations"
)

parser.add_argument('--mapper', nargs='+', type=int, default=[0],
      help="Mapper for Frequency Attention Module"
)

parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default='../logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')

args = parser.parse_args()

if args.seed == -1 :
    seed = np.random.randint(1e5)
else:
    seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
print("Random Seed:  " + str(seed))
if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id + '_' + str(seed)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

data_path = "../data"
dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None
    
dropouts = tuple(args.dropout for i in range(3))

model = None
regularizer = None
print(args.model)
print("Emb Reg: " + args.regularizer + '  ' +str(args.reg))
print("Time Reg: " + str(args.reg_t))
print("Core Reg: " + str(args.reg_w))
if dataset.Tag == False:
    exec('model = '+args.model+'(dataset.get_shape(), dropouts, args.rank1, args.init)')
else:
    exec('model = '+args.model+'(dataset.get_shape(), dropouts, args.rank1, args.rank2, args.init, args.ratio, args.no_time_emb, args.regularizer, args.mapper)')
exec('regularizer = '+args.regularizer+'(args.reg)')

device = 'cuda'
model.to(device)
assert args.reg_t >=0, "Invalid time reg weight."
assert args.reg_w >=0, "Invalid core reg weight."

regularizer = [regularizer, TimeReg(args.reg_t, args.p), CoreReg(args.reg_w)]
for reg in regularizer:
    reg.to(device)
    
optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)

if args.checkpoint != '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))

head_entity = [14010]   #<Tom_Adeyemi>
relation = [43,42]      # <playsFor><occursSince> , <playsFor><occursUntil>
tail_entity = [10337, 1985, 8193, 11876]
# <Oldham_Athletic_A.F.C.> 11-12 <Brentford_F.C.> 12-13 <Leeds_United_F.C.> 15-16 <Rotherham_United_F.C.> 16-17
timestamp = [161,162,163,164,165,166,167]
# 2011-2017

data = []
for i in head_entity:
    for k in tail_entity:    
        for l in timestamp:
            for j in relation:
                data.append([i,j,k,l])
data = torch.from_numpy(np.array(data)).cuda()
head_entity = torch.from_numpy(np.array(head_entity)).cuda()
relation = torch.from_numpy(np.array(relation)).cuda()
tail_entity = torch.from_numpy(np.array(tail_entity)).cuda()
timestamp = torch.from_numpy(np.array(timestamp)).cuda()

value = torch.zeros(len(head_entity), len(tail_entity), len(relation)*len(timestamp))

model.eval()
scores, _ = model.forward(data)
scores = scores.view(len(head_entity), len(tail_entity), len(relation)*len(timestamp), scores.size(1))
print(scores.size())

for i in range(len(head_entity)):
    for k in range(len(tail_entity)):    
        for l in range(len(timestamp)):
            for j in range(len(relation)):
                value[i,k,2*l+j] = scores[i,k,2*l+j,tail_entity[k]]
print(value)
np.save('value.npy', value.cpu().detach().numpy())
att_weight = model.get_att_weight(relation)
print(att_weight[0])
print(att_weight[1])
np.save('att_weight.npy', att_weight.cpu().detach().numpy())

