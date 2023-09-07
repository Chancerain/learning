import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

BATCH_SIZE = 128


# 1.准备数据集
def get_dataloader(train=True):
    transforms_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = MNIST(root='./data', train=train, transform=transforms_fn)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader


# 2.构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        # 1.修改形状
        x = input.view([input.size(0), 1 * 28 * 28])
        # 2.进行全连接操作
        x = self.fc1(x)
        # 3.激活函数的处理
        x = F.relu(x)
        # 3.输出层
        out = self.fc2(x)

        return F.log_softmax(out, dim=-1)


model = MnistModel()
model.load_state_dict(torch.load("./model/model.pkl"))
optimizer = Adam(model.parameters(), lr=0.001)
optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))


# 3.实现训练过程
def train(epoch):
    data_loader = get_dataloader()
    for idx, (input, target) in enumerate(data_loader):
        optimizer.zero_grad()  # 梯度归零
        output = model(input)  # 调用模型，得到预测值
        loss = F.nll_loss(output, target)  # 得到损失
        loss.backward()  # 反向传播
        optimizer.step()  # 梯度更新

        if idx % 100 == 0:
            print(epoch, idx, loss.item())

        # 模型的保存
        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")


train(1)
