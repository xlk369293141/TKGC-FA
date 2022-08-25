from abc import ABC, abstractmethod
from turtle import forward
from typing import Tuple

import torch
from torch import nn
from ConLoss import SupContrastive_Loss

class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class Fro(Regularizer):
    def __init__(self, weight: float):
        super(Fro, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.norm(f, 2) ** 2
                )
        return norm / factors[0][0].shape[0]

class L2(Regularizer):
    def __init__(self, weight: float):
        super(L2, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 2
                )
        return norm / factors[0][0].shape[0]

class L1(Regularizer):
    def __init__(self, weight: float):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f)**1
                )
        return norm / factors[0][0].shape[0]

class NA(Regularizer):
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        return torch.Tensor([0.0]).cuda()

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += self.weight * torch.sum(
                    torch.abs(f) ** 3
                ) / f.shape[0]
        return norm
    
class DURA(Regularizer):
    def __init__(self, weight: float):
        super(DURA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0

        for factor in factors:
            h, r, t = factor

            norm += torch.sum(t**2 + h**2)
            norm += torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]

class DURA_W(Regularizer):
    def __init__(self, weight: float):
        super(DURA_W, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            h, r, t = factor

            norm += 0.5 * torch.sum(t**2 + h**2)
            norm += 1.5 * torch.sum(h**2 * r**2 + t**2 * r**2)

        return self.weight * norm / h.shape[0]
    
class ConR(Regularizer):
    def __init__(self, weight: float):
        super(ConR, self).__init__()
        self.weight = weight
        self.con_loss = SupContrastive_Loss(tau = 1.0)
        
    def forward(self, factors):
        norm = 0
        for factor in factors:
            q, t, q_label, t_label = factor
            norm += self.con_loss(q_label, q)
            norm += self.con_loss(t_label, t)
        return self.weight * norm / t.shape[0]

class T_DURA(Regularizer):
    def __init__(self, weight: float):
        super(T_DURA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0

        for factor in factors:
            h, r, t, time_diff = factor
            
            # print(time_diff)
            # print(torch.sum(time_diff))
            norm += torch.sum(time_diff)
            # print(norm)
            # norm += torch.sum(t**2 + h**2) 
            # print(norm)
            # norm += torch.sum(h**2 * r**2 + t**2 * r**2)
            # norm += torch.sum(h**2 * T**2 + t**2 * T**2 + r**2 * T**2)
            # print(norm)
        #print(norm)

        return self.weight * torch.sqrt(torch.sqrt(norm + 1e-8))/ h.shape[0]   

