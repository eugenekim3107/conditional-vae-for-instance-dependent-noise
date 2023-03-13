import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm


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



