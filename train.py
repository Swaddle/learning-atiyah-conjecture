import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn import CrossEntropyLoss 
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

from learning_atiyah import PolyM, SimpleLinear

from functools import reduce 
from itertools import cycle 

import os 

from strong_checkpoints import Checkpointer

def one_hot(tensor):
    max_index = torch.argmax(tensor.abs())
    one_hot = torch.zeros_like(tensor)
    one_hot[max_index] = 1
    return one_hot


def gen_random_sample_2d(n_points: int):
    p = torch.randn(n_points,2)
    v = torch.randn(n_points)
    dots = torch.empty(n_points)

    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) -  p.unsqueeze(0)
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:,:,0]

    Xi = torch.stack(( (M + xs).sqrt() , (M - xs).sqrt())) 
    coeff_tensors = []
    for j in range(n_points):
        poly_j = []
        for k in range(n_points): 
            if j==k:
                continue 
            else:
                y_jk = ps[j,k][1]
                if y_jk < 0:
                    poly_j.append(PolyM([-Xi[1][j,k], Xi[0][j,k]]))
                elif y_jk > 0:
                    poly_j.append(PolyM([ Xi[0][j,k], Xi[1][j,k]]))
                else:  # y_jk =0 
                    x_jk = ps[j,k][0]
                    if x_jk < 0:
                        poly_j.append(PolyM([Xi[0][j,k], Xi[1][j,k]]))
                    else:
                        poly_j.append(PolyM([Xi[0][j,k], Xi[1][j,k]]))
        
        prod_poly_j = reduce((lambda x, y: x * y), poly_j).values()
        coeffs = torch.stack(prod_poly_j)        
        dots[j] = torch.dot(coeffs, v) 
        coeff_tensors.append(coeffs)
    
    sample = torch.stack(coeff_tensors)
    classification = one_hot(dots) 
    return sample, classification 

def gen_batch(n_points: int, bz: int): 
    inpts = []
    targets = []
    for k in range(bz):
        sample, cls = gen_random_sample_2d(n_points)
        inpts.append(sample.flatten()) 
        targets.append(cls) 
    return torch.stack(inpts), torch.stack(targets)

def train():
    num_iters = 10000 
    n_points = 4
    input_dim = n_points ** 2
    
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])

    model = SimpleLinear(input_dim, n_points, 1024)
    
    criterion = CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.00005)
    
    if local_rank==0:
        checkpointer = Checkpointer({"model": model.state_dict()})
    
    data =  cycle((gen_batch(n_points,32) for k in range(1000))) 

    for k, (inpt, target) in enumerate(data):
        outpt = model(inpt)
        loss = criterion(outpt, target)
        loss.backward()
        optimizer.step()
        
        if local_rank == 0:    
            if k%500==0:
                checkpointer.save()

        writer.add_scalar("Loss/train", loss, k)

if __name__ == "__main__":
    train()

writer.flush()
