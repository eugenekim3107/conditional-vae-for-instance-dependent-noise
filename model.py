import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.out = nn.Linear(1568, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x

class IDNGenerator:
    def __init__(self, X, y, p, epochs, model, optimizer, loss_fn, data_loader):
        self.X = X
        self.y = y
        self.p = p
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.loss_fn = loss_fn

    def train(self):
        self.model.train()
        loop = tqdm(self.data_loader, leave=True)
        correct = 0
        total = 0
        mean_loss = []
        for batch_idx, (x, y) in enumerate(loop):
            out = self.model(x)
            loss = self.loss_fn(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            mean_loss.append(loss.item())
            correct += int(sum(out.argmax(axis=1) == y))
            total += y.size(0)

            loop.set_postfix(loss=loss.item())

        mean_losses = sum(mean_loss) / len(mean_loss)
        accu = 100. * (correct / total)
        print(f"Loss: {mean_losses}, Accuracy: {accu}")
        self.model.eval()
        return self.model(self.X)

    def generate(self):
        S_list = []
        for epoch in range(self.epochs):
            S_t = self.train()
            S_list.append(S_t)
        S = sum(S_list) / len(S_list)
        one_hot = torch.zeros((self.y.size(0), self.y.max() + 1))
        one_hot[np.arange(self.y.size(0)), self.y] = True
        S[one_hot.to(torch.bool)] = float('-inf')
        N = torch.amax(S, dim=1)
        y_noise = torch.argmax(S, dim=1)
        num_noise = int(self.p * N.size(0))
        indices = torch.sort(N)[1][:num_noise]
        y_final = self.y.clone().detach()
        for i in indices:
            y_final[i] = y_noise[i]
        return self.X, y_final

'''
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

class VAE(nn.Module):
    def __init__(self, x_dim):

def main():
    model = ResNet18()
    x = torch.randn((5, 1, 28, 28))
    print(model(x))

if __name__ == "__main__":
    main()



