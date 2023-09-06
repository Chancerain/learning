import torchvision
from torch import nn
from torch.utils.data import DataLoader

vgg16_true = torchvision.models.vgg16(pretrained=True)

train_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                          download=True)

dataloader = DataLoader(train_data, batch_size=64)

vgg16_true.add_module('add_linear', nn.Linear(1000, 10))

print(vgg16_true)