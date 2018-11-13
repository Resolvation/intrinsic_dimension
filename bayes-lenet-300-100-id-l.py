import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim

from core.data_loader import mnist_data
from core.evaluate import evaluate
from core.layers import LinearOffset, NetWrapper
from core.train import stdn_loss, train
from core.utils import save


def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Bayes_LeNet_300_100_ID_L(450, 250, 50).to(device)
    train_loader, test_loader = mnist_data()
    criterion = stdn_loss(net, len(train_loader.dataset))
    optimizer = optim.Adam(net.parameters())

    # net.load_state_dict(torch.load("./logs/Bayes_LeNet_300_100_ID_L.tar"))
    train(device, net, train_loader, criterion, optimizer)

    """
    print(torch.exp(net.log_sigma_sqr1[:5] / 2))
    print(torch.mean(torch.exp(net.log_sigma_sqr1 / 2)))
    print(torch.median(torch.exp(net.log_sigma_sqr1 / 2)))
    print(torch.exp(net.log_sigma_sqr2[:5] / 2))
    print(torch.mean(torch.exp(net.log_sigma_sqr2 / 2)))
    print(torch.median(torch.exp(net.log_sigma_sqr2 / 2)))
    print(torch.exp(net.log_sigma_sqr3[:5] / 2))
    print(torch.mean(torch.exp(net.log_sigma_sqr3 / 2)))
    print(torch.median(torch.exp(net.log_sigma_sqr3 / 2)))
    """

    print("Trainset")
    evaluate(device, net, train_loader, criterion, num_nets=10)

    print("Testset")
    criterion = stdn_loss(net, len(test_loader.dataset))
    evaluate(device, net, test_loader, criterion, num_nets=10)

    save(net, "Bayes_LeNet_300_100_ID_L")


class Bayes_LeNet_300_100_ID_L(NetWrapper):
    """
    Neural network with two linear layers of size 300 and 100 in joint intristic
    dimention of size d.
    """
    def __init__(self, d1, d2, d3):
        super().__init__()
        setattr(self, "train", True)
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.register_parameter("mu1", Parameter(torch.zeros(d1)))
        self.register_parameter("log_sigma_sqr1", Parameter(torch.full(tuple([d1]), -12)))
        self.register_parameter("mu2", Parameter(torch.zeros(d2)))
        self.register_parameter("log_sigma_sqr2", Parameter(torch.full(tuple([d2]), -12)))
        self.register_parameter("mu3", Parameter(torch.zeros(d3)))
        self.register_parameter("log_sigma_sqr3", Parameter(torch.full(tuple([d3]), -12)))
        self.l1 = LinearOffset(d1, 28*28, 300)
        self.l2 = LinearOffset(d2, 300, 100)
        self.l3 = LinearOffset(d3, 100, 10)

    def forward(self, x):
        if self.train:
            eps1 = torch.randn_like(self.log_sigma_sqr1)
            eps2 = torch.randn_like(self.log_sigma_sqr2)
            eps3 = torch.randn_like(self.log_sigma_sqr3)
        else:
            eps1 = 0.
            eps2 = 0.
            eps3 = 0.
        theta_d1 = self.mu1 + eps1 * (1e-16 + self.log_sigma_sqr1.exp()).sqrt()
        theta_d2 = self.mu2 + eps2 * (1e-16 + self.log_sigma_sqr2.exp()).sqrt()
        theta_d3 = self.mu3 + eps3 * (1e-16 + self.log_sigma_sqr3.exp()).sqrt()
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, theta_d1))
        x = F.relu(self.l2(x, theta_d2))
        return self.l3(x, theta_d3)

    def KL(self):
        return ((self.mu1**2+self.log_sigma_sqr1.exp()-self.log_sigma_sqr1).sum()+(self.mu2**2+self.log_sigma_sqr2.exp()-self.log_sigma_sqr2).sum()+(self.mu3**2+self.log_sigma_sqr3.exp()-self.log_sigma_sqr3).sum()-self.d1-self.d2-self.d3) / 2


if __name__ == "__main__":
    main()
