import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.data_loader import mnist_data
from core.evaluate import evaluate
from core.layers import BayesLinear, NetWrapper
from core.train import stdn_loss, train
from core.utils import save

def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Bayes_LeNet_300_100().to(device)
    train_loader, test_loader = mnist_data()
    criterion = stdn_loss(net, len(train_loader.dataset))
    optimizer = optim.Adam(net.parameters())

    #net.load_state_dict(torch.load("./logs/Bayes_LeNet_300_100.tar"))
    train(device, net, train_loader, criterion, optimizer)

    """
    print(torch.exp(net.l1.log_sigma_sqr[:5, :5] / 2))
    print(torch.mean(torch.exp(net.l1.log_sigma_sqr / 2)))
    print(torch.median(torch.exp(net.l1.log_sigma_sqr / 2)))
    print(torch.exp(net.l2.log_sigma_sqr[:5, :5] / 2))
    print(torch.mean(torch.exp(net.l2.log_sigma_sqr / 2)))
    print(torch.median(torch.exp(net.l2.log_sigma_sqr / 2)))
    print(torch.exp(net.l3.log_sigma_sqr[:5, :5] / 2))
    print(torch.mean(torch.exp(net.l3.log_sigma_sqr / 2)))
    print(torch.median(torch.exp(net.l3.log_sigma_sqr / 2)))
    """

    print("Trainset")
    evaluate(device, net, train_loader, criterion, num_nets=10)

    print("Testset")
    criterion = stdn_loss(net, len(test_loader.dataset))
    evaluate(device, net, test_loader, criterion, num_nets=10)

    save(net, "Bayes_LeNet_300_100")


class Bayes_LeNet_300_100(NetWrapper):
    """
    Neural network with two linear layers of size 300 and 100.
    """
    def __init__(self):
        super().__init__()
        setattr(self, "train", True)
        self.l1 = BayesLinear(28*28, 300)
        self.l2 = BayesLinear(300, 100)
        self.l3 = BayesLinear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

    def KL(self):
        return sum([child.KL() for child in self.children()])


if __name__ == "__main__":
    main()
