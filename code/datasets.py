import pickle
from typing import Dict, Tuple, List
import os

import numpy as np
import torch
from models import KBCModel

class Dataset(object):
    def __init__(self, data_path: str, name: str):
        self.root = os.path.join(data_path, name)
        print(name)
        TKGC = ['ICEWS14', 'ICEWS05-15', 'GDELT', 'YAGO15K']
        if name in TKGC:
            self.Tag = True
        else:
            self.Tag = False
        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
            self.data[f] = pickle.load(in_file)
        # for f in ['train', 'test1', 'test2']:
        #     in_file = open(os.path.join(self.root, f + '.pickle'), 'rb')
        #     self.data[f] = pickle.load(in_file)

        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        self.n_predicates *= 2
        if maxis.shape[0] > 4:
            self.n_timestamps = max(int(maxis[3] + 1), int(maxis[4] + 1))
        else:
            self.n_timestamps = int(maxis[3] + 1)
        inp_f = open(os.path.join(self.root, 'to_skip.pickle'), 'rb')
        if self.Tag == True:
            self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
        else:
            self.to_skip: Dict[str, Dict[Tuple[int, int], List[int]]] = pickle.load(inp_f)
        inp_f.close()
        
        print("Num of Entity/Relation/Timestamp", self.get_shape())
        print("Train set:\t", self.data['train'].shape)

    def get_weight(self):
        appear_list = np.zeros(self.n_entities)
        copy = np.copy(self.data['train'])
        for triple in copy:
            if self.Tag == False:
                h, r, t = triple
            else:
                h, r, t, tau = triple
            appear_list[h] += 1
            appear_list[t] += 1

        w = appear_list / np.max(appear_list) * 0.9 + 0.1
        return w

    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  # has been multiplied by two.
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
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.n_predicates // 2
            ranks = model.get_ranking(q, self.to_skip[m], batch_size=10000)

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
        # if self.Tag == True:            
        #     Years = self.data['train'][:,3]
        #     Months = self.data['train'][:,4]
        #     Days = self.data['train'][:,5]
        #     datelist = []
        #     for i in range(len(Years)):
        #         datelist.append(datetime.date(Years[i], Months[i], Days[i]))
        #     oldest = min(datelist)
        #     youngest = max(datelist)
        #     print("The start date:", oldest, "\tThe end date:", youngest)
        #     delta = relativedelta.relativedelta(youngest, oldest)#日期差
        #     print("Time Span: \t", delta.years, 'Years,', delta.months, 'months,', delta.days, 'days')
        return self.n_entities, self.n_predicates, self.n_timestamps
    
    def calculate(self):
        #distance dic    fact:time dic
        time_diff = {}
        fact_time = {}

        facts = []
        for i in ['train','valid','test']:
            for item in self.data[i]:
                facts = [item[0], item[1], item[2]]
                if facts in fact_time:
                    diff = item[3] - fact_time[facts]
                    if diff in time_diff:
                        time_diff[diff] += 1
                        # fact_time[facts] = item[3]
                    else:
                        time_diff[diff] = 1
                        # fact_time[facts] = item[3]
                    fact_time[facts] = item[3]
        return time_diff
            
        
        
        