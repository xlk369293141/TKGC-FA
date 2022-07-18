from abc import ABC, abstractmethod
from typing import Tuple, List, Dict
import numpy as np
import torch
from torch import nn

from tqdm import tqdm

class KBCModel(nn.Module, ABC):
    def get_ranking(
            self, queries: torch.Tensor,
            filters: Dict[Tuple[int, int], List[int]],
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
                        filter_out = filters[(query[0].item(), query[1].item())]
                        filter_out += [queries[b_begin + i, 2].item()]   # Add the tail of this (b_begin + i) query
                        scores[i, torch.LongTensor(filter_out)] = -1e6
                    ranks[b_begin:b_begin + batch_size] += torch.sum(
                        (scores >= targets).float(), dim=1
                    ).cpu()
                    b_begin += batch_size
                    bar.update(batch_size)
        return ranks

class ComplEx_con(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(ComplEx_con, self).__init__()
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
                   (torch.cat(lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0], dim=1),
                    torch.cat(rhs[0], rhs[1], dim=1), x[:, 2], x[:,0]*self.sizes(1)+x[:,1])
               ]

class TuckER_con(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3
    ):
        super(TuckER_con, self).__init__()
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
                   (query, rhs, x[:, 2], x[:,0]*self.sizes(1)+x[:,1])
               ]

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
                   (query, rhs, x[:, 2], x[:,0]*self.sizes(1)+x[:,1])
               ]
        
class ComplEx_DE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-, ratio: float = 0.5
    ):
        super(ComplEx_DE, self).__init__()
        self.sizes = sizes
        self.rank = rank
        self.init_size = init_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, 2 * rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size
        
        self.t_emb_dim = int(ratio * rank)
        
        self.create_time_embedds()
        self.activate_func = torch.sin  # torch.exp

    def create_time_embedds(self):
        self.m_freq = nn.Embedding(self.sizes[0], 2 * self.t_emb_dim)
        self.d_freq = nn.Embedding(self.sizes[0], 2 * self.t_emb_dim)
        self.y_freq = nn.Embedding(self.sizes[0], 2 * self.t_emb_dim)

        self.m_freq.weight.data *= self.init_size
        self.d_freq.weight.data *= self.init_size
        self.y_freq.weight.data *= self.init_size
                                
        self.m_phi = nn.Embedding(self.sizes[0], 2 * self.t_emb_dim)
        self.d_phi = nn.Embedding(self.sizes[0], 2 * self.t_emb_dim)
        self.y_phi = nn.Embedding(self.sizes[0], 2 * self.t_emb_dim)

        self.m_phi.weight.data *= self.init_size
        self.d_phi.weight.data *= self.init_size
        self.y_phi.weight.data *= self.init_size
    
    def get_time_embedd(self, entities, year, month, day):
        y = self.activate_func(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.activate_func(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.activate_func(self.d_freq(entities)*day + self.d_phi(entities))
        
        pad_dim = self.rank - self.t_emb_dim
        pad_emb = torch.ones([self.size[0], pad_dim])
        time_emb = torch.cat((pad_emb, (y+m+d)[:, :self.t_emb_dim], pad_emb, (y+m+d)[:, self.t_emb_dim:]), 1)
        return time_emb
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs = lhs * self.get_time_embedd(x[:, 0], x[:,3], x[:,4], x[:,5])
        rhs = rhs * self.get_time_embedd(x[:, 2], x[:,3], x[:,4], x[:,5])
        
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

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
                               
class TuckER_DE(KBCModel):
    def __init__(
            self, sizes: Tuple[int, int, int], rank: int,
            init_size: float = 1e-3, ratio: float = 0.5
    ):
        super(TuckER_DE, self).__init__()
        self.sizes = sizes
        self.rank = rank

        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (rank, rank, rank)), 
                                    dtype=torch.float, device="cuda", requires_grad=True))
        # self.W.data *= init_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(s, rank, sparse=True)
            for s in sizes[:2]
        ])
        self.embeddings[0].weight.data *= init_size
        self.embeddings[1].weight.data *= init_size

        self.t_emb_dim = int(ratio * rank)
        
        self.create_time_embedds()
        self.activate_func = torch.sin  # torch.exp
        
    def create_time_embedds(self):
        self.m_freq = nn.Embedding(self.sizes[0], self.t_emb_dim)
        self.d_freq = nn.Embedding(self.sizes[0], self.t_emb_dim)
        self.y_freq = nn.Embedding(self.sizes[0], self.t_emb_dim)

        self.m_freq.weight.data *= self.init_size
        self.d_freq.weight.data *= self.init_size
        self.y_freq.weight.data *= self.init_size
                                
        self.m_phi = nn.Embedding(self.sizes[0], self.t_emb_dim)
        self.d_phi = nn.Embedding(self.sizes[0], self.t_emb_dim)
        self.y_phi = nn.Embedding(self.sizes[0], self.t_emb_dim)

        self.m_phi.weight.data *= self.init_size
        self.d_phi.weight.data *= self.init_size
        self.y_phi.weight.data *= self.init_size
    
    def get_time_embedd(self, entities, year, month, day):
        y = self.activate_func(self.y_freq(entities)*year + self.y_phi(entities))
        m = self.activate_func(self.m_freq(entities)*month + self.m_phi(entities))
        d = self.activate_func(self.d_freq(entities)*day + self.d_phi(entities))
        
        pad_dim = self.rank - self.t_emb_dim
        pad_emb = torch.ones([self.size[0], pad_dim])
        time_emb = torch.cat((pad_emb, (y+m+d)[:, :self.t_emb_dim]), 1)
        return time_emb
    
    def forward(self, x):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])
        lhs = lhs * self.get_time_embedd(x[:, 0], x[:,3], x[:,4], x[:,5])
        rhs = rhs * self.get_time_embedd(x[:, 2], x[:,3], x[:,4], x[:,5])
        
        query = lhs.view(-1, 1, lhs.size(1))

        W_mat = torch.mm(rel, self.W.view(rel.size(1), -1))
        W_mat = W_mat.view(-1, lhs.size(1), lhs.size(1))

        query = torch.bmm(query, W_mat)
        query = query.view(-1, lhs.size(1))      
        
        to_score = self.embeddings[0].weight
        return (
                    torch.mm(query, to_score.transpose(1,0))
                ), [
                   (query, rhs, x[:, 2], x[:,0]*self.sizes(1)+x[:,1])
               ]