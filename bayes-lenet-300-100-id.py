import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.data_loader import mnist_data
from core.evaluate import evaluate
from core.layers import StochasticModule, LinearOffset
from core.train import stdn_loss, train
from core.utils import save


def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Bayes_LeNet_300_100_ID(750).to(device)
    train_loader, test_loader = mnist_data()
    criterion = stdn_loss(model, len(train_loader.dataset))
    test_criterion = stdn_loss(model, len(test_loader.dataset))
    optimizer = optim.Adam(model.parameters())

    train(device, model, train_loader, test_loader,
          criterion, optimizer, test_criterion)

    save(model, "Bayes_LeNet_300_100_ID")


class Bayes_LeNet_300_100_ID(StochasticModule):
    """
    Neural modelwork with two linear layers of size 300 and 100 in joint intristic
    dimention of size d.
    """
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.register_parameter("mu", nn.Parameter(torch.zeros(d)))
        self.register_parameter("log_sigma_sqr",
                                nn.Parameter(torch.full(tuple([d]), -12)))
        self.l1 = LinearOffset(d, 28*28, 300)
        self.l2 = LinearOffset(d, 300, 100)
        self.l3 = LinearOffset(d, 100, 10)

    def forward(self, x):
        if self.stochastic:
            eps = torch.randn_like(self.log_sigma_sqr)
        else:
            eps = 0.
        theta_d = self.mu + eps * (1e-16 + self.log_sigma_sqr.exp()).sqrt()
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x, theta_d))
        x = F.relu(self.l2(x, theta_d))
        return self.l3(x, theta_d)

    def KL(self):
        return (self.mu ** 2 + self.log_sigma_sqr.exp() \
                - self.log_sigma_sqr).sum() / 2 - self.d


if __name__ == "__main__":
    main()
