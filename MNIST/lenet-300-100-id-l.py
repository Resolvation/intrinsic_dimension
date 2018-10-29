import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import mnist_data_loaders, train, test, save
from layers import LinearOffsetLayer


def main():
    torch.manual_seed(6717449005)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = mnist_data()

    model = Lenet_300_100_ID_L(350, 350, 50).to(device)
    optimizer = optim.Adam(model.parameters())

    train(model, device, train_loader, criterion, optimizer)
    test(model, device, test_loader, criterion)
    save(model)


class Lenet_300_100_ID_L(nn.Module):
    '''Neural network with two linear layers of size 200 with layerwise intrinsic dimentions.'''
    def __init__(self, d1, d2, d3):
        super().__init__()
        self.register_parameter('theta_d1', torch.nn.Parameter(torch.zeros(d1)))
        self.register_parameter('theta_d2', torch.nn.Parameter(torch.zeros(d2)))
        self.register_parameter('theta_d3', torch.nn.Parameter(torch.zeros(d3)))
        self.lof1 = LinearOffsetLayer(d1, 28*28, 300)
        self.lof2 = LinearOffsetLayer(d2, 300, 100)
        self.lof3 = LinearOffsetLayer(d3, 100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.lof1(x, self.theta_d1))
        x = F.relu(self.lof2(x, self.theta_d2))
        y_pred = self.lof3(x, self.theta_d3)
        return y_pred


if __name__ == '__main__':
    main()
