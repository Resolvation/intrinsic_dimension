import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import mnist_data, train, test, save
from layers import LinearOffsetLayer


def main():
    torch.manual_seed(6717449005)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = mnist_data()

    model = Lenet_300_100_ID(750).to(device)
    optimizer = optim.Adam(model.parameters())

    train(model, device, train_loader, criterion, optimizer)
    test(model, device, test_loader, criterion)
    save(model)


class Lenet_300_100_ID(nn.Module):
    '''Neural network with two linear layers of size 200 in intristic dimention of size d.'''
    def __init__(self, d):
        super().__init__()
        self.register_parameter('theta_d', torch.nn.Parameter(torch.zeros(d)))
        self.lof1 = LinearOffsetLayer(d, 28*28, 300)
        self.lof2 = LinearOffsetLayer(d, 300, 100)
        self.lof3 = LinearOffsetLayer(d, 100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.lof1(x, self.theta_d))
        x = F.relu(self.lof2(x, self.theta_d))
        y_pred = self.lof3(x, self.theta_d)
        return y_pred


if __name__ == '__main__':
    main()
