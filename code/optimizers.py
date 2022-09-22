import tqdm
import torch
from torch import nn
from torch import optim

from models import KBCModel
from regularizers import *


class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: list, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer[0]
        self.regularizer_t = regularizer[1]
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        
    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)
        
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size
                ].cuda()
                
                predictions, factors = self.model.forward(input_batch)
                assert not torch.any(torch.isnan(factors[0][0])), "nan"
                assert not torch.any(torch.isnan(factors[0][1])), "nan"
                assert not torch.any(torch.isnan(factors[0][2])), "nan"
                assert not torch.any(torch.isnan(factors[0][3])), "nan"
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg, l_w = self.regularizer.forward(factors, self.model.W)
                l_w = 0.01 * l_w
                l_t = torch.zeros_like(l_reg)
                if self.regularizer_t is not None:
                    if self.model.no_time_emb:
                        l_t = self.regularizer_t.forward(self.model.embeddings[2].weight[:-1])
                    else:
                        l_t = self.regularizer_t.forward(self.model.embeddings[2].weight)
                # l_w = torch.Tensor([0.0]).cuda()
                l = l_fit + l_reg + l_t + l_w

                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.1f}', reg=f'{l_reg.item():.1f}', reg_t=f'{l_t.item():.2f}', reg_w=f'{l_w.item():.2f}')
