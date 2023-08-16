import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss 

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

def gen_batch_cpu(n_points: int, bz: int): 
    inpts = []
    targets = []
    for k in range(bz):
        sample, cls = gen_random_sample_2d(n_points)
        inpts.append(sample.flatten()) 
        targets.append(cls) 
    return torch.stack(inpts), torch.stack(targets)


def train():
    n_points = 4
    input_dim = n_points ** 2 
    save_path = "/mnt/Client/strongcompute_michael/checkpoints/latest.pt"
    local_model = SimpleLinear(input_dim, n_points, 1024)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        local_model.load_state_dict(checkpoint["model"])
        local_model.eval() 

    criterion = CrossEntropyLoss() 
    data = (gen_batch_cpu(n_points,32) for _ in range(1000))
    loss_av = 0

    for k, (inpt, target) in enumerate(data):
        outpt = local_model(inpt)
        loss = criterion(outpt, target)
        loss_av = loss_av + loss
        print(loss_av/k)

if __name__ == "__main__":
    train()

