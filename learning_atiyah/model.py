from torch.nn import Linear, Module
from torch.nn.functional import relu, log_softmax

class SimpleLinear(Module):
    def __init__(self, in_dim: int, out_dim: int, d: int):
        super(SimpleLinear, self).__init__()  
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.l1 = Linear(in_dim, d)
        self.l2 = Linear(d,d)
        self.l3 = Linear(d,d)
        self.l4 = Linear(d,d)
        self.l5 = Linear(d, out_dim) 

    def forward(self,x):
        x = relu(self.l1(x))
        x = relu(self.l2(x))
        x = relu(self.l3(x))
        x = relu(self.l4(x))
        x = self.l5(x)
        return x

