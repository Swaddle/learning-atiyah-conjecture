import torch.distributed as dist


def lr_lambda(initial_lr, step):
    factor = 0.01
    return initial_lr / (1 + factor * step)


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
