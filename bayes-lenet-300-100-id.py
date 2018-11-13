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
    net = Bayes_LeNet_300_100_ID(750).to(device)
    train_loader, test_loader = mnist_data()
    criterion = stdn_loss(net, len(train_loader.dataset))
    optimizer = optim.Adam(net.parameters())

    # net.load_state_dict(torch.load("./logs/Bayes_LeNet_300_100_ID.tar"))
    train(device, net, train_loader, criterion, optimizer)

    """
    print(torch.exp(net.log_sigma_sqr[:5] / 2))
    print(torch.mean(torch.exp(net.log_sigma_sqr / 2)))
    print(torch.median(torch.exp(net.log_sigma_sqr / 2)))
    """

    print("Trainset")
    evaluate(device, net, train_loader, criterion, num_nets=10)

    print("Testset")
    criterion = stdn_loss(net, len(test_loader.dataset))
    evaluate(device, net, test_loader, criterion, num_nets=10)

    save(net, "Bayes_LeNet_300_100_ID")


class Bayes_LeNet_300_100_ID(NetWrapper):
    """
    Neural network with two linear layers of size 300 and 100 in joint intristic
    dimention of size d.
    """
    def __init__(self, d):
        super().__init__()
        setattr(self, "train", True)
        self.d = d
        self.register_parameter("mu", Parameter(torch.zeros(d)))
        self.register_parameter("log_sigma_sqr", Parameter(torch.full(tuple([d]), -12)))
        self.l1 = LinearOffset(d, 28*28, 300)
        self.l2 = LinearOffset(d, 300, 100)
        self.l3 = LinearOffset(d, 100, 10)

    def forward(self, x):
        if self.train:
            eps = torch.randn_like(self.log_sigma_sqr)
        else:
            eps = 0.
        theta_d = self.mu + eps * (1e-16 + self.log_sigma_sqr.exp()).sqrt()
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, theta_d))
        x = F.relu(self.l2(x, theta_d))
        return self.l3(x, theta_d)

    def KL(self):
        return ((self.mu**2 + self.log_sigma_sqr.exp() - self.log_sigma_sqr).sum()
               - self.d) / 2


if __name__ == "__main__":
    main()
