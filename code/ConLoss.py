import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupContrastive_Loss(nn.Module):

    def __init__(self, tau=0.5, sim_kernel='dot'):
        super(SupContrastive_Loss, self).__init__()
        self.tau = tau
        self.sim_kernel = sim_kernel

    def similarity(self, x1, x2):
        # Dot Product Kernel
        M = torch.mm(x1, x2.transpose(1, 0))/self.tau
        if self.sim_kernel == 'dot':
            s = torch.exp(M - torch.max(M, dim=1, keepdim=True)[0])
        elif self.sim_kernel == 'cosine':
            tmp = M/x1.norm(dim=-1, p=2).unsqueeze(-1)
            tmp = tmp/x2.norm(dim=-1, p=2).unsqueeze(-2)
            s = torch.exp(tmp)
        return s

    def forward(self, batch_label, *x):
        X = torch.cat(x, 0)
        batch_labels = torch.cat([batch_label for i in range(len(x))], 0)
        len_ = batch_labels.size()[0]

        # computing similarities for each positive and negative pair
        s = self.similarity(X, X)

        # computing masks for contrastive loss
        if len(x)==1:
            mask_i = torch.from_numpy(np.ones((len_, len_))).to(batch_labels.device)
        else:
            mask_i = 1. - torch.from_numpy(np.identity(len_)).to(batch_labels.device) # sum over items in the numerator
        label_matrix = batch_labels.unsqueeze(0).repeat(len_, 1)
        mask_j = (batch_labels.unsqueeze(1) - label_matrix == 0).float()*mask_i # sum over items in the denominator
        pos_num = torch.sum(mask_j, 1)

        # weighted NLL loss
        s_i = torch.clamp(torch.sum(s*mask_i, 1), min=1e-10) 
        s_j = torch.clamp(s*mask_j, min=1e-10)
        del mask_i
        log_p = torch.sum(-torch.log(s_j/s_i)*mask_j, 1)/pos_num
        del mask_j
        loss = torch.mean(log_p)

        return loss