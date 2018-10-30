import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from layers import BayesLinear, NetWrapper
from utils import mnist_data, train, test, evaluate, save


def main():
    torch.manual_seed(6717449005)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = mnist_data()

    model = Lenet_300_100().to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = my_loss(model, len(train_loader.dataset))

    train(model, device, train_loader, criterion, optimizer, n_epochs=400)

    model.set_flag('train', False)
    test(model, device, train_loader, criterion)

    model.set_flag('train', True)
    evaluate(model, device, train_loader, criterion, 10)

    criterion = my_loss(model, len(test_loader.dataset))

    model.set_flag('train', False)
    print('Mean')
    test(model, device, test_loader, criterion)

    model.set_flag('train', True)
    print('Stochastic')
    evaluate(model, device, test_loader, criterion, 10)
    save(model)


class Lenet_300_100(NetWrapper):
    '''Neural network with two linear layers of size 200.'''

    def __init__(self):
        super().__init__()
        setattr(self, 'train', True)
        self.linear1 = BayesLinear(28*28, 300)
        self.linear2 = BayesLinear(300, 100)
        self.linear3 = BayesLinear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = self.linear3(x)
        return y_pred

    def KL(self):
        res = 0
        for child in self.children():
            res += child.KL()
        return res


def my_loss(model, len_dataset):
    def loss(outputs, labels):
        return F.cross_entropy(outputs, labels) + model.KL()/len_dataset
    return loss


if __name__ == '__main__':
    main()
