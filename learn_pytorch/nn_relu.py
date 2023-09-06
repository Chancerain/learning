import torch
import torchvision.datasets
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)


class DM(nn.Module):
    def __init__(self):
        super(DM, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output


dm = DM()

step = 0

writer = SummaryWriter("../logs_relu")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = dm(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()
