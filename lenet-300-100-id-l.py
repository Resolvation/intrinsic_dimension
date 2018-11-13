import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.data_loader import mnist_data
from core.evaluate import evaluate
from core.layers import LinearOffset
from core.train import train
from core.utils import save


def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = LeNet_300_100_ID_L(450, 250, 50).to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = mnist_data()
    optimizer = optim.Adam(net.parameters())

    train(device, net, train_loader, criterion, optimizer)

    print("Trainset:")
    evaluate(device, net, train_loader, criterion)
    print("Testset:")
    evaluate(device, net, test_loader, criterion)

    save(net, "LeNet_300_100_ID_L")


class LeNet_300_100_ID_L(nn.Module):
    """
    Neural network with two linear layers of size 300 and 100 in intristic
    dimentions of size d1, d2, d3.
    """
    def __init__(self, d1, d2, d3):
        super().__init__()
        self.register_parameter("theta_d1", nn.Parameter(torch.zeros(d1)))
        self.register_parameter("theta_d2", nn.Parameter(torch.zeros(d2)))
        self.register_parameter("theta_d3", nn.Parameter(torch.zeros(d3)))
        self.l1 = LinearOffset(d1, 28*28, 300)
        self.l2 = LinearOffset(d2, 300, 100)
        self.l3 = LinearOffset(d3, 100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, self.theta_d1))
        x = F.relu(self.l2(x, self.theta_d2))
        return self.l3(x, self.theta_d3)


if __name__ == "__main__":
    main()
