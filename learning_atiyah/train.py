import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from poly import PolyM
from functools import reduce 

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

    for j in range(n_points):
        poly_j = []
        for k in range(n_points): 
            if j==k:
                next 
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
        
        prod = reduce((lambda x, y: x * y), poly_j)
        
        coeff_tensor = torch.stack(prod.values())
        print(coeff_tensor)

if __name__ == "__main__":
    train()
