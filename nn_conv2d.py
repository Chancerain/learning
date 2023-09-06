import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class DM(nn.Module):
    def __init__(self):
        super(DM, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


dm = DM()

writer = SummaryWriter("../logs")

step = 0

for data in dataloader:
    imgs, targets = data
    output = dm(imgs)
    # print(imgs.shape)
    # print(output.shape)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step += 1
