import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learning.learn_pytorch.model import *

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

dm = DM()

loss_fn = nn.CrossEntropyLoss()

learn_rate = 1e-2
optimizer = torch.optim.SGD(dm.parameters(), lr=learn_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

writer = SummaryWriter("../logs_train")

for i in range(epoch):
    print("第{}轮训练开始".format(i + 1))

    for data in train_dataloader:
        imgs, targets = data
        outputs = dm(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    total_test_loss = 0
    total_accruracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = dm(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accruracy += accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accruracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accruracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(dm, "dm_{}.pth".format(i))
    print("模型已保存")

writer.close()
