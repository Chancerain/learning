import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class DM(nn.Module):
    def __init__(self):
        super(DM, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output


dm = DM()

for data in dataloader:
    imgs, targets = data
    output = torch.flatten(imgs)
    print(output.shape)
    output = dm(output)
    print(output.shape)
