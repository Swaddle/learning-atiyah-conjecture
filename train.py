from os import replace, environ
from functools import partial
from itertools import islice, cycle

import torch
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import CrossEntropyLoss

from learning_atiyah import (
    SimpleLinear,
    all_reduce_grads,
    gen_batch,
    lr_lambda,
    multiply_grads,
)

# writer = SummaryWriter()

from pathlib import Path


def train():
    dim = 2
    n_points = 4
    dual_dim = n_points
    batch_size = 32
    num_samples = 500
    model_d = 2048
    num_epochs = 100
    initial_lr = 0.005

    # input is the list [(x_1,y_1)...(x_n,y_n)] + dual vector

    input_dim = n_points * dim + dual_dim
    # = (dim + 1) * n_points
    
    save_path_str = f"/mnt/Client/strongcompute_michael/checkpoints/latest_{model_d}_{dim}_{n_points}_{num_samples}_{batch_size}.pt"
    latest_path = Path(save_path_str)
    temp_path = Path(save_path_str+".tmp")

    dist.init_process_group(backend="nccl")
    torch.manual_seed(0)

    local_rank = int(environ["LOCAL_RANK"])
    world_size = dist.get_world_size()

    local_model = SimpleLinear(input_dim, n_points, model_d)
    local_model.to(device=local_rank)

    lr_lambda_func = partial(lr_lambda, initial_lr)
    criterion = CrossEntropyLoss()
    local_optimizer = torch.optim.SGD(local_model.parameters(), lr=initial_lr)
    local_scheduler = lr_scheduler.LambdaLR(local_optimizer, lr_lambda_func)

    current_sample = 0
    current_epoch = 0

    if latest_path.is_file():
        with latest_path.open("rb") as f:
            checkpoint = torch.load(f)
        current_sample = checkpoint["checkpoint_sample"]
        current_epoch = checkpoint["checkpoint_epoch"]
        local_model.load_state_dict(checkpoint["local_model"])
        local_scheduler.load_state_dict(checkpoint["local_scheduler"])
        local_optimizer.load_state_dict(checkpoint["local_optimizer"])
        local_model.train()
    else: 
        latest_path.touch()  
    
    data = cycle(zip(range(num_samples), (gen_batch(n_points, batch_size, local_rank) for _ in range(num_samples))))
    
    num_samples_seen = 0

    for e in islice(range(num_epochs), current_epoch, num_epochs):
        for idx, (inpt, target) in islice(data, current_sample, num_samples):

            outpt = local_model(inpt)
            loss = criterion(outpt, target)
            loss.backward()

            all_reduce_grads(local_model)
            multiply_grads(local_model, 1.0 / world_size)

            local_optimizer.step()

            del inpt, target  

            num_samples_seen = num_samples_seen + 1 

            if (idx + 1) % 100 == 0:
                local_scheduler.step()

            if local_rank == 0:
                if idx % 50 == 0:
                    print({"loss":loss, "sample": idx, "epoch":e})

                if num_samples_seen % 200 == 0:
                    with temp_path.open("wb") as f:
                        torch.save( 
                            {
                                "local_model": local_model.state_dict(),
                                "local_scheduler": local_scheduler.state_dict(),
                                "local_optimizer": local_optimizer.state_dict(),
                                "checkpoint_sample": idx,
                                "checkpoint_epoch": e,
                            },
                            f,
                        )
                        f.flush()
                    replace(temp_path, latest_path)
            #writer.add_scalar("Loss/train", loss, k)

        # reset to continue iter
        current_sample = 0



if __name__ == "__main__":
    train()

# writer.flush()
