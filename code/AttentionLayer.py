import math
import torch
import torch.nn as nn

class TimeDCTBase(nn.Module):
    def __init__(self, mapper, length, channel):
        super(TimeDCTBase, self).__init__()
        assert channel % len(mapper) == 0
        self.num_heads = len(mapper)
        self.register_buffer('weight', self.get_dct_filter(mapper, length, channel))
        
    def forward(self, x):
        assert len(x.shape) == 2, 'x must been 2 dimensions, but got ' + str(len(x.shape))
        
        x = x.T * self.weight
        result = torch.sum(x, dim=1)
        return result.T
                
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, mapper, tile_size, channel):
        # mapper: the set of chosen frequcy, len(mapper) = num_heads
        # tile_size: the length of time sequence
        # channel: the num of time embedding dimension
        dct_filter = torch.zeros(channel, tile_size)

        c_part = channel // len(mapper)

        for i, u_x in enumerate(mapper):
            for t_k in range(tile_size):
                dct_filter[i * c_part: (i+1)*c_part, t_k] = self.build_filter(t_k, u_x, tile_size)
                        
        return dct_filter