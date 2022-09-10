from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import numpy as np
import torch
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
        
class ComplEx(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                       (lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                       (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1)
               ), [
                   (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]
               
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
                        
class ComplEx_DFT(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], dropouts: Tuple[float, float, float], rank1: int, rank2: int,
            init_size: float = 1e-3, ratio: float = 0.5
    ):
        super(ComplEx_DFT, self).__init__()
        self.sizes = sizes
        self.rank = rank1
        self.init_size = init_size
        self.t_emb_dim = int(ratio * self.rank)
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for (s, rank) in zip(sizes[:3], [2 * self.rank, 4*self.rank-2*self.t_emb_dim, 2 * self.t_emb_dim])
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        self.embeddings[2].weight.data *= init_size

        # self.W.data *= init_size
        
    def get_time_embedd(self, relations, timestamps):
        B = relations.size(0)
        assert(relations.size(1) == 4*self.rank-2*self.t_emb_dim)
        assert(timestamps.size(1) == 2*self.t_emb_dim)
        rel = relations.view(2, B,-1)
        tim = timestamps.view(2, B,-1)
        tmp = torch.cat((rel, tim), dim=2)
        tmp = tmp.view(2, B, self.rank, 2)
        tmp = tmp[:,:,:,0] * tmp[:,:,:,1]
        return tmp
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        tim = self.embeddings[2](x[:, 3])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        temporal_rel = self.get_time_embedd(rel, tim)
        temporal_rel = (temporal_rel[0, :, :], temporal_rel[1, :, :])

        to_score = self.embeddings[0].weight
        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return (
                    (lhs[0] * temporal_rel[0] - lhs[1] * temporal_rel[1]) @ to_score[0].transpose(0, 1) +
                    (lhs[0] * temporal_rel[1] + lhs[1] * temporal_rel[0]) @ to_score[1].transpose(0, 1)
                ), [
                   (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                    torch.sqrt(temporal_rel[0] ** 2 + temporal_rel[1] ** 2),
                    torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))
               ]
                
class TuckER_DFT(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], dropouts: Tuple[float, float, float], rank1: int, rank2: int,
            init_size: float = 1e-3, ratio: float = 0.5
    ):
        super(TuckER_DFT, self).__init__()
        self.sizes = sizes
        self.rank1 = rank1
        self.rank2 = rank2
        self.init_size = init_size
        self.t_emb_dim = int(ratio * rank2)
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank1, rank2, rank1)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        # nn.init.xavier_uniform_(self.W)
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
    
    def get_time_embedd(self, relations, timestamps):
        B = relations.size(0)
        tmp = torch.cat((relations, timestamps), 1)
        tmp = tmp.view(B, self.rank2, 2)
        temporal_relation_emb = tmp[:,:,0] * tmp[:,:,1]
        return temporal_relation_emb
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        tim = self.embeddings[2](x[:, 3])
        
        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, lhs.size(1))
        
        temporal_rel = self.get_time_embedd(rel, tim)
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
                   (torch.sqrt(lhs ** 2), torch.sqrt(temporal_rel ** 2 + 1e-10), torch.sqrt(rhs ** 2))
               ]
                
class TuckER_ATT(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], dropouts: Tuple[float, float, float], rank1: int, rank2: int,
            init_size: float = 1e-3, ratio: float = 0.5
    ):
        super(TuckER_ATT, self).__init__()
        self.sizes = sizes
        self.rank1 = rank1
        self.rank2 = rank2
        self.init_size = init_size
        self.t_emb_dim = int(ratio * rank2)
        
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

        mapper = [0,1]  # mapper can be a list with [0,1,2,3,4]
        mapper = [temp * (sizes[2] // 5) for temp in mapper] 
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
        B = relations.size(0)
        dct_base = self.dct_layer(T_emb)
        dct_base = relations[:,self.rank2-self.t_emb_dim: self.rank2] * dct_base.view(1, D)
        dct_att = self.fc(dct_base)
        tmp = timestamps * dct_att.expand_as(timestamps)
        tmp = torch.cat((relations, tmp), 1)
        tmp = tmp.view(B, self.rank2, 2)
        temporal_relation_emb = tmp[:,:,0] * tmp[:,:,1]
        return temporal_relation_emb
    
    def forward(self, y):
        lhs = self.embeddings[0](y[:, 0])
        rel = self.embeddings[1](y[:, 1])
        rhs = self.embeddings[0](y[:, 2])
        tim = self.embeddings[2](y[:, 3])

        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, lhs.size(1))
        
        # Calculate the temporal relation embedding using attention module
        temporal_rel = self.get_time_embedd(rel, tim)
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
                   (torch.sqrt(lhs ** 2), torch.sqrt(temporal_rel ** 2 + 1e-10), torch.sqrt(rhs ** 2))
               ]