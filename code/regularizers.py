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
    
class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            for f in factor:
                assert not torch.any(torch.isnan(f)), "nan factor"
                norm += torch.sum(
                    torch.abs(f) ** 3
                )
        return self.weight * norm / factors[0][0].shape[0]
        
class TmpReg(Regularizer):
    def __init__(self, weight: float):
        super(TmpReg, self).__init__()
        self.weight = weight

    def forward(self, factors):
        norm = 0
        for factor in factors:
            lhs, rel_t, rhs = factor
            assert not torch.any(torch.isnan(lhs)), "nan lhs"
            assert not torch.any(torch.isnan(rel_t)), "nan rel_t"
            assert not torch.any(torch.isnan(rhs)), "nan rhs"
            lhs = lhs.view(-1, 1, lhs.size(1))
            q1 = torch.bmm(lhs, rel_t)
            rhs = rhs.view(-1, 1, rhs.size(1))
            q2 = torch.bmm(rhs, rel_t)
            assert(q1.size() == q2.size())
            
            norm +=  torch.sum(q1**2 + rhs**2)
            norm +=  torch.sum(lhs**2 + q2**2)

        return self.weight * norm / factors[0][0].shape[0]

class TimeReg(Regularizer):
    def __init__(self, weight: float, p: int):
        super(TimeReg, self).__init__()
        self.weight = weight
        self.p = p
        
    def forward(self, tim):
        assert not torch.any(torch.isnan(tim)), "nan time embedding"
        time_diff = torch.diff(tim, dim=0)
        norm = torch.sum(time_diff.abs() ** self.p)
        return self.weight * norm / time_diff.shape[0]
    
class CoreReg(Regularizer):
    def __init__(self, weight: float):
        super(CoreReg, self).__init__()
        self.weight = weight
        
    def forward(self, W):
        assert not torch.any(torch.isnan(W)), "nan core tensor"
        e = torch.ones_like(W)
        for i in range(W.size(0)):
            e[i,i,i] = 0.0
        w_norm = torch.sum((W * e).abs() ** 4)
        return self.weight * w_norm
    
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