

import numpy as np
import torch

def BMIncrements(B, dim=1, N=100, T=1):
    # generate brownian increment
    dat = torch.randn(B, N, dim)*np.sqrt(T/N)
    
    return torch.cat([dat, -dat], dim=0)


'''
def augment(x, T=1):
    # add time dimension to each path
    """
    x -- input tensor of shape (batch, length, dim)
    """
    B, N, dim = x.size()
    time = torch.tensor([T/N * i for i in range(1, N+1)], dtype=torch.float64)
    augmented = torch.zeros(B, N, dim+1)
    for i in range(B):
        augmented[i][:][0] = time
        augmented[i][:][1:] = x[i]
    return augmented
    '''