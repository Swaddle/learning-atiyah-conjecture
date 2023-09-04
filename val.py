import os

import torch
from torch.nn import CrossEntropyLoss

from learning_atiyah import SimpleLinear, gen_small_batch_Sn_perm, gen_batch


def val():
    n_points = 4
    dim = 2
    input_dim = n_points * (dim + 1)
    save_path = "/mnt/Client/strongcompute_michael/checkpoints/latest_64_2_4_500_16.pt"
    local_model = SimpleLinear(input_dim, n_points, 64)

    if os.path.isfile(save_path):
        checkpoint = torch.load(save_path)
        local_model.load_state_dict(checkpoint["local_model"])
        local_model.eval()

    criterion = CrossEntropyLoss()
    data = (gen_batch(n_points,8,'cpu') for _ in range(1000))

    for k, (inpt, target) in enumerate(data):
        outpt = local_model(inpt)
        loss = criterion(outpt, target)
        print(loss)


if __name__ == "__main__":
    val()
