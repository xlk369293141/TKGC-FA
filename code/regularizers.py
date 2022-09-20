from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn
from ConLoss import SupContrastive_Loss

class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factors: Tuple[torch.Tensor]):
        pass

class NA(Regularizer):
    def __init__(self, weight: float):
        super(NA, self).__init__()
        self.weight = weight

    def forward(self, factors):
        return torch.Tensor([0.0]).cuda()
    
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
                norm += torch.sum(
                    torch.abs(f) ** 2
                )
        return self.weight * norm / factors[0][0].shape[0]

class L1(Regularizer):
    def __init__(self, weight: float):
        super(L1, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += torch.sum(
                    torch.abs(f)**1
                )
        return self.weight * norm / factors[0][0].shape[0]

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += torch.sum(
                    torch.abs(f) ** 3
                )
        return self.weight * norm / factors[0][0].shape[0]

class L4(Regularizer):
    def __init__(self, weight: float):
        super(L4, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                norm += torch.sum(
                    torch.norm(f, 4) ** 4
                )
        return self.weight * norm / factors[0][0].shape[0]
    
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
    
class TmpReg(Regularizer):
    def __init__(self, weight: float):
        super(TmpReg, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            lhs, rel_t, rhs = factor
            lhs = lhs.view(-1, 1, lhs.size(1))
            q1 = torch.bmm(lhs, rel_t)
            rhs = rhs.view(-1, 1, rhs.size(1))
            q2 = torch.bmm(rhs, rel_t)
            assert(q1.size() == q2.size())
            
            norm +=  0.5 * torch.sum(q1**2 + rhs**2)
            norm +=  0.5 * torch.sum(lhs**2 + q2**2)

        return self.weight * norm / lhs.shape[0]
    
class TimeReg(Regularizer):
    def __init__(self, weight: float, p: int):
        super(TimeReg, self).__init__()
        self.weight = weight
        self.p = p
        
    def forward(self, tim):
        assert not torch.any(torch.isnan(tim)), "nan tim"
        norm = 0
        time_diff = torch.diff(tim, dim=0)
        for f in time_diff:
            norm += torch.sum(
                    torch.norm(f, self.p) ** self.p
                )
            
        return self.weight * norm / time_diff.shape[0]
    
class CoreReg(Regularizer):
    def __init__(self, weight: float, p: int):
        super(CoreReg, self).__init__()
        self.weight = weight
        self.p = p
        
    def forward(self, W):
        assert not torch.any(torch.isnan(W)), "nan tim"
        norm = 0
        norm += torch.norm(W, self.p) ** self.P
        return self.weight * norm
    
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