from torch.nn import Linear, Module
from torch.nn.functional import relu


class SimpleLinear(Module):
    def __init__(self, in_dim: int, out_dim: int, d: int):
        super(SimpleLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.l1 = Linear(in_dim, d)
        self.l2 = Linear(d, 2 * d)
        self.l3 = Linear(2 * d, 4 * d)
        self.l4 = Linear(4 * d, 2 * d)
        self.l5 = Linear(2 * d, out_dim)

    def forward(self, x):
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = relu(self.l3(x))
        x = relu(self.l4(x))
        x = self.l5(x)
        return x
