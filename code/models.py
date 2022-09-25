from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import numpy as np
import torch
import math
from AttentionLayer import TimeDCTBase
from torch import nn
from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters,
            batch_size: int = 1000, chunk_size: int = -1
    ):
        """
        Returns filtered ranking for each queries.
        :param queries: a torch.LongTensor of triples (lhs, rel, rhs)
        :param filters: filters[(lhs, rel)] gives the rhs to filter from ranking
        :param batch_size: maximum number of queries processed at once
        :return:
        """
        ranks = torch.ones(len(queries))
        with tqdm(total=queries.shape[0], unit='ex') as bar:
            bar.set_description(f'Evaluation')
            with torch.no_grad():
                b_begin = 0
                while b_begin < len(queries):
                    these_queries = queries[b_begin:b_begin + batch_size]
                    target_idxs = these_queries[:, 2].cpu().tolist()
                    scores, _ = self.forward(these_queries)
                    targets = torch.stack([scores[row, col] for row, col in enumerate(target_idxs)]).unsqueeze(-1)
                    assert not torch.any(torch.isinf(scores)), "inf scores"
                    assert not torch.any(torch.isnan(scores)), "nan scores"
                    assert not torch.any(torch.isinf(targets)), "inf targets"
                    assert not torch.any(torch.isnan(targets)), "nan targets"
                    for i, query in enumerate(these_queries):
                        if len(list(filters.keys())[0]) == 2:
                            filter_out = filters[(query[0].item(), query[1].item())]
                        elif len(list(filters.keys())[0]) == 3:
                            filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]
                        else:
                            raise ValueError("The Query Component should be 2 or 5, but got ", len(list(filters.keys())[0]))
                        filter_out += [queries[b_begin + i, 2].item()]   # Add the tail of this (b_begin + i) query
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks
    
    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()
               
class TuckER(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(TuckER, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank, rank, rank)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        # self.W *= init_size
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        query = lhs.view(-1, 1, lhs.size(1))

        W_mat = torch.mm(rel, self.W.view(rel.size(1), -1))
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))

        query = torch.bmm(query, W_mat) 
        query = query.view(-1, lhs.size(1))      
        
        to_score = self.embeddings[0].weight
        return (
                    torch.mm(query, to_score.transpose(1,0))
                ), [
                   (torch.sqrt(lhs ** 2), torch.sqrt(rel ** 2), torch.sqrt(rhs ** 2))
               ]
                
class TuckER_DFT(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], dropouts: Tuple[float, float, float], rank1: int, rank2: int,
            init_size: float = 1e-2, ratio: float = 0.0, no_time_emb=False
    ):
        super(TuckER_DFT, self).__init__()
        self.sizes = sizes
        self.rank1 = rank1
        self.rank2 = rank2
        self.init_size = init_size
        self.no_time_emb = no_time_emb

        self.t_emb_dim = int(ratio * rank2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-init_size, init_size, (rank1, rank2, rank1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))

        self.bn0 = torch.nn.BatchNorm1d(rank1)
        self.bn1 = torch.nn.BatchNorm1d(rank1)
        self.input_dropout = torch.nn.Dropout(dropouts[0])
        self.hidden_dropout1 = torch.nn.Dropout(dropouts[1])
        self.hidden_dropout2 = torch.nn.Dropout(dropouts[2])
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for (s, rank) in zip(sizes[:3], [rank1, 2*rank2-self.t_emb_dim, self.t_emb_dim])
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        tim = self.embeddings[2](x[:, 3])
        
        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, lhs.size(1))
        
        tmp_r = torch.cat((rel, tim), 1)
        tmp_r1, tmp_r2 = tmp_r[:, :self.rank2], tmp_r[:, self.rank2:]
        temporal_rel = tmp_r1 * tmp_r2
        temporal_rel = torch.mm(temporal_rel, self.W.view(temporal_rel.size(1), -1))
        temporal_rel = temporal_rel.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(temporal_rel)
        
        x = torch.bmm(x, W_mat)
        x = x.view(-1, lhs.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        to_score = self.embeddings[0].weight
        return (
                    torch.mm(x, to_score.transpose(1,0))
                ), [
                   (torch.sqrt(lhs ** 2), torch.sqrt(tmp_r1 ** 2 + tmp_r2 ** 2 + 1e-8), torch.sqrt(rhs ** 2))
               ]
                
class TuckER_ATT(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], dropouts: Tuple[float, float, float], rank1: int, rank2: int,
            init_size: float = 1e-3, ratio: float = 0.5, no_time_emb=False, reg='N3'
    ):
        super(TuckER_ATT, self).__init__()
        self.sizes = sizes
        self.rank1 = rank1
        self.rank2 = rank2
        self.init_size = init_size
        self.no_time_emb = no_time_emb

        self.t_emb_dim = int(ratio * rank2)
        self.reg = reg
        
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank1, rank2, rank1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        
        # self.W.data *= init_size
        self.bn0 = torch.nn.BatchNorm1d(rank1)
        self.bn1 = torch.nn.BatchNorm1d(rank1)
        self.input_dropout = torch.nn.Dropout(dropouts[0])
        self.hidden_dropout1 = torch.nn.Dropout(dropouts[1])
        self.hidden_dropout2 = torch.nn.Dropout(dropouts[2])
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for (s, rank) in zip(sizes[:3], [rank1, 2*rank2-self.t_emb_dim, self.t_emb_dim])
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size
        mapper = [0]  # mapper can be a list with [0-12]
        mapper = [temp * (sizes[2] // 12) for temp in mapper] 
        print("Mapper Freq: " + str(mapper))
        self.num_heads = len(mapper)
        self.dct_layer = TimeDCTBase(mapper, sizes[2], self.t_emb_dim)
        reduction = 4
        self.fc = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.t_emb_dim // reduction, self.t_emb_dim, bias=False),
            nn.Sigmoid()
        )
        
    def get_time_embedd(self, relations, timestamps):
        # self.register_buffer('weight', self.get_freq_att(height, width, mapper_x, mapper_y, channel))
        # self.register_parameter('weight', self.get_freq_att(height, width, mapper_x, mapper_y, channel))
        # att = self.get_freq_att(self.embeddings[2].weight) # [1, self.rank2]
        T_emb = self.embeddings[2].weight
        A, D = T_emb.size()
        dct_base = self.dct_layer(T_emb)
        dct_base = relations[:,self.rank2-self.t_emb_dim: self.rank2] * dct_base.view(1, D)
        dct_att = self.fc(dct_base)
        tmp = timestamps * dct_att.expand_as(timestamps)
        tmp = torch.cat((relations, tmp), 1)

        return tmp[:, :self.rank2], tmp [:, self.rank2:]
    
    def forward(self, y):
        lhs = self.embeddings[0](y[:, 0])
        rel = self.embeddings[1](y[:, 1])
        rhs = self.embeddings[0](y[:, 2])
        tim = self.embeddings[2](y[:, 3])

        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, lhs.size(1))
        
        # Calculate the temporal relation embedding using attention module
        tmp_r1, tmp_r2 = self.get_time_embedd(rel, tim)
        temporal_rel = tmp_r1 * tmp_r2
        temporal_rel = torch.mm(temporal_rel, self.W.view(temporal_rel.size(1), -1))
        temporal_rel = temporal_rel.view(-1, lhs.size(1), lhs.size(1))
        W_mat = self.hidden_dropout1(temporal_rel)
        
        x = torch.bmm(x, W_mat)
        x = x.view(-1, lhs.size(1))      
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        to_score = self.embeddings[0].weight
        if self.reg == 'TmpReg':
            regular = [(lhs, temporal_rel, rhs)]
        else:
            regular = [(math.pow(2, 1 / 3)  *lhs, tmp_r1, tmp_r2, math.pow(2, 1 / 3) * rhs)]
        return (
                    torch.mm(x, to_score.transpose(1,0))
                ), regular
                
# class TNTComplEx(KBCModel):
#     def __init__(
#             self, sizes: Tuple[int, int, int, int], dropouts: Tuple[float, float, float], rank1: int, rank2: int, 
#              init_size: float = 1e-2, ratio: float = 0.5, no_time_emb=False,
#     ):
#         super(TNTComplEx, self).__init__()
#         self.sizes = sizes
#         self.rank = rank1

#         self.embeddings = nn.ModuleList([
#             nn.Embedding(s, 2 * rank1, sparse=True)
#             for s in [sizes[0], sizes[1], sizes[2], sizes[1]]  # last embedding modules contains no_time embeddings
#         ])
#         self.embeddings[0].weight.data *= init_size
#         self.embeddings[1].weight.data *= init_size
#         self.embeddings[2].weight.data *= init_size
#         self.embeddings[3].weight.data *= init_size

#         self.no_time_emb = no_time_emb

#     @staticmethod
#     def has_time():
#         return True

#     def score(self, x):
#         lhs = self.embeddings[0](x[:, 0])
#         rel = self.embeddings[1](x[:, 1])
#         rel_no_time = self.embeddings[3](x[:, 1])
#         rhs = self.embeddings[0](x[:, 2])
#         time = self.embeddings[2](x[:, 3])

#         lhs = lhs[:, :self.rank], lhs[:, self.rank:]
#         rel = rel[:, :self.rank], rel[:, self.rank:]
#         rhs = rhs[:, :self.rank], rhs[:, self.rank:]
#         time = time[:, :self.rank], time[:, self.rank:]
#         rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

#         rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
#         full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

#         return torch.sum(
#             (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) * rhs[0] +
#             (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) * rhs[1],
#             1, keepdim=True
#         )

#     def forward(self, x):
#         lhs = self.embeddings[0](x[:, 0])
#         rel = self.embeddings[1](x[:, 1])
#         rel_no_time = self.embeddings[3](x[:, 1])
#         rhs = self.embeddings[0](x[:, 2])
#         time = self.embeddings[2](x[:, 3])

#         lhs = lhs[:, :self.rank], lhs[:, self.rank:]
#         rel = rel[:, :self.rank], rel[:, self.rank:]
#         rhs = rhs[:, :self.rank], rhs[:, self.rank:]
#         time = time[:, :self.rank], time[:, self.rank:]
#         rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

#         right = self.embeddings[0].weight
#         right = right[:, :self.rank], right[:, self.rank:]

#         rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
#         rrt = rt[0] - rt[3], rt[1] + rt[2]
#         full_rel = rrt[0] + rnt[0], rrt[1] + rnt[1]

#         regularizer = (
#            math.pow(2, 1 / 3) * torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
#            torch.sqrt(rrt[0] ** 2 + rrt[1] ** 2),
#            torch.sqrt(rnt[0] ** 2 + rnt[1] ** 2),
#            math.pow(2, 1 / 3) * torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2)
#         )
#         return (
#                (lhs[0] * full_rel[0] - lhs[1] * full_rel[1]) @ right[0].t() +
#                (lhs[1] * full_rel[0] + lhs[0] * full_rel[1]) @ right[1].t()
#             ), [regularizer]
        

#     def forward_over_time(self, x):
#         lhs = self.embeddings[0](x[:, 0])
#         rel = self.embeddings[1](x[:, 1])
#         rhs = self.embeddings[0](x[:, 2])
#         time = self.embeddings[2].weight

#         lhs = lhs[:, :self.rank], lhs[:, self.rank:]
#         rel = rel[:, :self.rank], rel[:, self.rank:]
#         rhs = rhs[:, :self.rank], rhs[:, self.rank:]
#         time = time[:, :self.rank], time[:, self.rank:]

#         rel_no_time = self.embeddings[3](x[:, 1])
#         rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

#         score_time = (
#             (lhs[0] * rel[0] * rhs[0] - lhs[1] * rel[1] * rhs[0] -
#              lhs[1] * rel[0] * rhs[1] + lhs[0] * rel[1] * rhs[1]) @ time[0].t() +
#             (lhs[1] * rel[0] * rhs[0] - lhs[0] * rel[1] * rhs[0] +
#              lhs[0] * rel[0] * rhs[1] - lhs[1] * rel[1] * rhs[1]) @ time[1].t()
#         )
#         base = torch.sum(
#             (lhs[0] * rnt[0] * rhs[0] - lhs[1] * rnt[1] * rhs[0] -
#              lhs[1] * rnt[0] * rhs[1] + lhs[0] * rnt[1] * rhs[1]) +
#             (lhs[1] * rnt[1] * rhs[0] - lhs[0] * rnt[0] * rhs[0] +
#              lhs[0] * rnt[1] * rhs[1] - lhs[1] * rnt[0] * rhs[1]),
#             dim=1, keepdim=True
#         )
#         return score_time + base

#     def get_rhs(self, chunk_begin: int, chunk_size: int):
#         return self.embeddings[0].weight.data[
#                chunk_begin:chunk_begin + chunk_size
#                ].transpose(0, 1)

#     def get_queries(self, queries: torch.Tensor):
#         lhs = self.embeddings[0](queries[:, 0])
#         rel = self.embeddings[1](queries[:, 1])
#         rel_no_time = self.embeddings[3](queries[:, 1])
#         time = self.embeddings[2](queries[:, 3])

#         lhs = lhs[:, :self.rank], lhs[:, self.rank:]
#         rel = rel[:, :self.rank], rel[:, self.rank:]
#         time = time[:, :self.rank], time[:, self.rank:]
#         rnt = rel_no_time[:, :self.rank], rel_no_time[:, self.rank:]

#         rt = rel[0] * time[0], rel[1] * time[0], rel[0] * time[1], rel[1] * time[1]
#         full_rel = (rt[0] - rt[3]) + rnt[0], (rt[1] + rt[2]) + rnt[1]

#         return torch.cat([
#             lhs[0] * full_rel[0] - lhs[1] * full_rel[1],
#             lhs[1] * full_rel[0] + lhs[0] * full_rel[1]
#         ], 1)