import tqdm
import torch
from torch import nn
from torch import optim

from models import KBCModel
from regularizers import Regularizer


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

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()
        
    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
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
                truth = input_batch[:, 2]

                l_fit = loss(predictions, truth)
                l_reg = self.regularizer.forward(factors)
                if self.regularizer_t is None:
                    l_t = torch.Tensor([0.0]).cuda()
                    l = l_fit + l_reg
                else:
                    l_t = self.regularizer_t.forward(self.model.embeddings[2].weight)
                    l = l_fit + l_reg + l_t
                
                self.optimizer.zero_grad()
                l.backward()

                self.optimizer.step()
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])

                bar.set_postfix(loss=f'{l.item():.1f}', reg=f'{l_reg.item():.1f}', reg_t=f'{l_t.item():.2f}')
               
        return l
