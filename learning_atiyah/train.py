import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from poly import PolyM

def train():
    num_epochs = 10 
    n_points = 4

    # (x_i, y_i)
    p = torch.randn(n_points,2)
    
    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) -  p.unsqueeze(0)
    
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:,:,0]

    Xi = torch.stack(( (M + xs).sqrt() , (M - xs).sqrt())) 

if __name__ == "__main__":
    train()
