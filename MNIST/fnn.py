import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from functions import data_loaders, train, test, save


def main():
    torch.manual_seed(43)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = data_loaders()

    model = FNN().to(device)
    optimizer = optim.Adam(model.parameters())

    train(model, device, train_loader, criterion, optimizer)
    test(model, device, test_loader, criterion)
    save(model)


class FNN(nn.Module):
    '''Neural network with two linear layers of size 200.'''
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(28*28, 200)
        self.linear2 = nn.Linear(200, 200)
        self.linear3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = self.linear3(x)
        return y_pred


if __name__ == '__main__':
    main()
