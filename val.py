import os
from functools import reduce

import torch
from torch.nn import CrossEntropyLoss

from learning_atiyah import PolyM, SimpleLinear
from random import choice

def cat_input(p, v):
    v = v.unsqueeze(0).transpose(0, 1)
    return torch.cat((p, v), 1)


def one_hot(tensor):
    max_index = torch.argmax(tensor.abs())
    one_hot = torch.zeros_like(tensor)
    one_hot[max_index] = 1
    return one_hot


def gen_random_sample_2d(n_points: int):
    p = torch.randn(n_points, 2)
    v = torch.randn(n_points)
    dots = torch.empty(n_points)

    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) - p.unsqueeze(0)
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:, :, 0]

    Xi = torch.stack(((M + xs).sqrt(), (M - xs).sqrt()))
    coeff_tensors = []
    for j in range(n_points):
        poly_j = []
        for k in range(n_points):
            if j == k:
                continue
            else:
                y_jk = ps[j, k][1]
                if y_jk < 0:
                    poly_j.append(PolyM([-Xi[1][j, k], Xi[0][j, k]]))
                elif y_jk > 0:
                    poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))
                else:  # y_jk =0
                    x_jk = ps[j, k][0]
                    if x_jk < 0:
                        poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))
                    else:
                        poly_j.append(PolyM([Xi[0][j, k], Xi[1][j, k]]))

        prod_poly_j = reduce((lambda x, y: x * y), poly_j).values()
        coeffs = torch.stack(prod_poly_j)
        dots[j] = torch.dot(coeffs, v)

    smple = (p, v)
    classification = one_hot(dots)
    return smple, classification

def filtered_range_mod_n(start, stop, n):
    x = range(start,stop)
    x = filter(lambda x: x % n != 0, x)
    return list(x)



def Sn_cycle_point_aug(p,cls):
    # length
    n = p.shape[0]
    s = choice(filtered_range_mod_n(0, 100, n))
    p.roll(s, 0)
    cls.roll(s, 0)
    return (p, cls)


def gen_batch_cpu(n_points: int, bz: int):
    inpts = []
    targets = []
    for _ in range(bz):
        (p, v), cls = gen_random_sample_2d(n_points)
        smple = cat_input(p, v)
        inpts.append(smple)
        targets.append(cls)
    
    return torch.stack(inpts), torch.stack(targets)


def gen_small_batch_Sn_perm(n_points, batch_size=4):
    inpts = []
    targets = []
   
    (p0, v0), cls0 = gen_random_sample_2d(n_points)
    smple0 = cat_input(p0, v0)
   

    inpts.append(smple0.flatten())
    targets.append(cls0)

    for _ in range(4):
        (p, cls) = Sn_cycle_point_aug(p0, cls0)
        inpts.append(cat_input(p, v0).flatten())
        targets.append(cls)
    
    return torch.stack(inpts), torch.stack(targets)



def val():
    n_points = 4
    dim = 2
    input_dim = n_points * (dim + 1)
    save_path = "/mnt/Client/strongcompute_michael/checkpoints/latest.pt"
    local_model = SimpleLinear(input_dim, n_points, 1024)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        local_model.load_state_dict(checkpoint["model"])
        local_model.eval()

    criterion = CrossEntropyLoss()
    data = (gen_small_batch_Sn_perm(n_points) for _ in range(1000))

    for k, (inpt, target) in enumerate(data):
    
        outpt = local_model(inpt)
        loss = criterion(outpt, target)
        print(loss)


if __name__ == "__main__":
    val()
