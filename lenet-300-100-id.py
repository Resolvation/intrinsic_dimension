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
    net = LeNet_300_100_ID(750).to(device)
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = mnist_data()
    optimizer = optim.Adam(net.parameters())

    train(device, net, train_loader, criterion, optimizer)

    print("Trainset:")
    evaluate(device, net, train_loader, criterion)
    print("Testset:")
    evaluate(device, net, test_loader, criterion)

    save(net, "LeNet_300_100_ID")


class LeNet_300_100_ID(nn.Module):
    """
    Neural network with two linear layers of size 300 and 100 in joint intristic
    dimention of size d.
    """
    def __init__(self, d):
        super().__init__()
        self.register_parameter("theta_d", torch.nn.Parameter(torch.zeros(d)))
        self.l1 = LinearOffset(d, 28*28, 300)
        self.l2 = LinearOffset(d, 300, 100)
        self.l3 = LinearOffset(d, 100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, self.theta_d))
        x = F.relu(self.l2(x, self.theta_d))
        return self.l3(x, self.theta_d)


if __name__ == "__main__":
    main()
