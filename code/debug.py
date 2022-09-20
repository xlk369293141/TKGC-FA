import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__() 
        
    def forward(self, x):
        x1 = x
        x2 = x+1
        x3 = x+3
        return [(x+1, x+2, x+3)]