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

def gen_batch(n_points: int, bz: int, device_id: int): 
    inpts = []
    targets = []
    for k in range(bz):
        sample, cls = gen_random_sample_2d(n_points)
        inpts.append(sample.flatten()) 
        targets.append(cls) 
    return torch.stack(inpts).to(device=device_id), torch.stack(targets).to(device=device_id)


def all_reduce_params(module):
    for param in module.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)

def multiply_params(module, mult_fact):
    for param in module.parameters():
        param.data = param.data * mult_fact

def all_reduce_grads(module):
    for param in module.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

def multiply_grads(module, mult_fact):
    for param in module.parameters():
        param.grad.data = param.grad.data * mult_fact

def train():
    num_iters = 100000 
    n_points = 4
    input_dim = n_points ** 2
    
    save_path = "/mnt/Client/strongcompute_michael/checkpoints/latest.pt"

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    local_model = SimpleLinear(input_dim, n_points, 1024)
    local_model.to(device=local_rank)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        local_model.load_state_dict(checkpoint["model"])
        local_model.train() 

    criterion = CrossEntropyLoss()
    local_optimizer = torch.optim.AdamW(local_model.parameters(),lr=0.00005)
    
    data =  cycle((gen_batch(n_points,32, local_rank) for k in range(1000))) 

    for k, (inpt, target) in enumerate(data):
        outpt = local_model(inpt)
        loss = criterion(outpt, target)
        loss.backward()

        all_reduce_grads(local_model)
        multiply_grads(local_model,1.0/world_size) 
        
        local_optimizer.step()
        

        if local_rank == 0:    
            if k%500==0:
                torch.save(
                    {
                        "model":local_model.state_dict()
                    },
                    save_path
                )

        writer.add_scalar("Loss/train", loss, k)

if __name__ == "__main__":
    train()

writer.flush()
