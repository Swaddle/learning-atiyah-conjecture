import os
from functools import reduce

import torch
from torch.nn import CrossEntropyLoss

from learning_atiyah import PolyM, SimpleLinear


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
        coeff_tensors.append(coeffs)

    sample = torch.stack(coeff_tensors)
    classification = one_hot(dots)
    return sample, classification


def Sn_cycle_point_aug(p, one_hot_class):
    # length
    n = p.shape[0]
    s = sample(filtered_range_mod_n(0, 100, n))
    p.roll(s, 0)
    one_hot_class(s, 0)
    return (p, one_hot_class)


def gen_batch_cpu(n_points: int, bz: int):
    inpts = []
    targets = []
    for k in range(bz):
        (p, v), cls = gen_random_sample_2d(n_points)
        sample = cat_input(p, v)
        inpts.append(sample)
        targets.append(cls)
    return torch.stack(inpts), torch.stack(targets)


def gen_small_batch_Sn_perm(n_points):
    inpts = []
    targets = []
    (p0, v0), cls0 = gen_random_sample_2d(n_points)
    inpts.append(sample)
    targets.append(cls)
    for k in range(4):
        (p, cls) = Sn_cycle_point_aug(p0, cls0)
        inpts.append(cat_input(p, v0))
        targets.append(cls)
    return torch.stack(inpts), torch.stack(targets)


def Sn_cycle_point_aug(p, one_hot_class):
    # length
    n = p.shape[0]
    s = sample(filtered_range_mod_n(0, 100, n))
    p.roll(s, 0)
    one_hot_class(s, 0)
    return (p, one_hot_class)


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
    data = (gen_small_batch_Sn_perm(n_points, 32) for _ in range(1000))

    for k, (inpt, target) in enumerate(data):
        outpt = local_model(inpt)
        loss = criterion(outpt, target)
        print(loss)


if __name__ == "__main__":
    train()
