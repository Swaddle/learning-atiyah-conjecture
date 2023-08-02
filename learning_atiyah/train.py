import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from poly import PolyM


# Simple Linear Model
class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def train():
    rank = dist.get_rank() % ( dist.world_size())
    model = nn.parallel.DistributedDataParallel(SimpleLinearModel(input_dim, output_dim).to(rank), device_ids=[rank])
    


    input_data, target_data = torch.randn(100, input_dim).to(rank), torch.randn(100, output_dim).to(rank)
    
    criterion, optimizer, num_epochs = nn.MSELoss(), optim.SGD(model.parameters(), lr=0.01), 10000
    

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        sleep(10)
        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()
        print(f"Rank {rank}, Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

if __name__ == "__main__":
    train()
