from functools import reduce
from random import choice

from torch import argmax, cat, dot, empty, randn, stack, zeros_like

from .poly import PolyM


def one_hot(tensor):
    max_index = argmax(tensor.abs())
    one_hot = zeros_like(tensor)
    one_hot[max_index] = 1
    return one_hot


def cat_input(p, v):
    v = v.unsqueeze(0).transpose(0, 1)
    return cat((p, v), 1)


def gen_random_sample_2d(n_points: int):
    p = randn(n_points, 2)
    v = randn(n_points)
    dots = empty(n_points)

    # differences
    # ps[j,k] = p[j] - p[k]
    ps = p.unsqueeze(1) - p.unsqueeze(0)
    # sum of x_ij^2 + y_ij^2
    M = ps.square().sum(2).sqrt()
    xs = ps[:, :, 0]

    Xi = stack(((M + xs).sqrt(), (M - xs).sqrt()))

    # coeff_tensors = []

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
        coeffs = stack(prod_poly_j)
        dots[j] = dot(coeffs, v)

        # coeff_tensors.append(coeffs)

    # sample = torch.stack(coeff_tensors)
    sample = (p, v)

    classification = one_hot(dots)
    return sample, classification


def gen_batch(n_points: int, bz: int, device_id: int):
    inpts = []
    targets = []
    for k in range(bz):
        (p, v), cls = gen_random_sample_2d(n_points)
        sample = cat_input(p, v)
        inpts.append(sample.flatten())
        targets.append(cls)
    return stack(inpts).to(device=device_id), stack(targets).to(device=device_id)


def filtered_range_mod_n(start: int, stop: int, n: int):
    x = range(start, stop)
    x = filter(lambda x: x % n != 0, x)
    return list(x)


def Sn_cycle_point_aug(p, cls):
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

    return stack(inpts), stack(targets)


def gen_small_batch_Sn_perm(n_points: int, batch_size: int = 4):
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

    return stack(inpts), stack(targets)
