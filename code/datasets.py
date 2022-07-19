import pickle
from typing import Dict, Tuple, List
import os

import numpy as np
import torch
from models import KBCModel

class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)

        print(self.data['train'].shape)
        self.convertTimes()

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2
        self.time = int(self.data['train'][3]+1)
        self.date = int(self.data['train'][4]+1)
        self.day = int(self.data['train'][5]+1)

        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()

    def convertTimes(self):      
        for split in ['train', 'valid', 'test']:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            h, r, t = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp1 = np.copy(copy[:, 0])
        tmp2 = np.copy(copy[:, 1])
        tmp3 = np.copy(copy[:, 2])
        copy[:, 0] = copy[:, 5]
        copy[:, 1] = copy[:, 4]
        copy[:, 2] = copy[:, 3] 
        copy[:, 3] = tmp3
        copy[:, 4] = tmp2
        copy[:, 5] = tmp1
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: KBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10), log_result=False, save_path=None
    ):
        model.eval()
        test = self.get_examples(split)
        examples = torch.from_numpy(test.astype('int64')).cuda()
        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}

        flag = False
        for m in missing:
            q = examples.clone()
            if n_queries > 0:
                permutation = torch.randperm(len(examples))[:n_queries]
                q = examples[permutation]
            if m == 'lhs':
                tmp1 = torch.clone(q[:, 0])
                tmp2 = torch.clone(q[:,1])
                tmp3 = torch.clone(q[:,1])
                q[:, 0] = q[:, 5]
                q[:, 5] = tmp1
                q[:, 1] = q[:, 4]
                q[:, 4] = tmp2
                q[:, 2] = q[:, 3]
                q[:, 3] = tmp3
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=500)

            if log_result:
                if not flag:
                    results = np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)
                    flag = True
                else:
                    results = np.concatenate((results, np.concatenate((q.cpu().detach().numpy(),
                                              np.expand_dims(ranks.cpu().detach().numpy(), axis=1)), axis=1)), axis=0)

            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                at
            ))))

        return mean_reciprocal_rank, hits_at

    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.time, self.date, self.day
