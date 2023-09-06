import torch
from torch import nn


class DM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


dm = DM()

x = torch.tensor(1.0)
output = dm(x)

print(dm)
