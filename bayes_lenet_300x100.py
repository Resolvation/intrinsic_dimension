import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from core.data_loader import mnist_data
from core.layers import StochasticModule, BayesLinear
from core.train import stdn_loss, train
from core.utils import save


def main():
    torch.manual_seed(42)

    writer = SummaryWriter("./logs/Bayes_LeNet_300_100")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Bayes_LeNet_300_100().to(device)

    train_loader, test_loader = mnist_data()
    criterion = stdn_loss(model, len(train_loader.dataset))
    test_criterion = stdn_loss(model, len(test_loader.dataset))
    optimizer = optim.Adam(model.parameters())

    train(writer, device, model, train_loader, test_loader,
          criterion, optimizer, test_criterion)

    writer.add_histogram("first layer sigmas",
                         model.l1.log_sigma_sqr.exp().sqrt().view(-1))
    writer.add_histogram("second layer sigmas",
                         model.l2.log_sigma_sqr.exp().sqrt().view(-1))
    writer.add_histogram("third layer sigmas",
                         model.l3.log_sigma_sqr.exp().sqrt().view(-1))

    save(model, "Bayes_LeNet_300_100")
    writer.close()


class Bayes_LeNet_300_100(StochasticModule):
    """
    Neural modelwork with two linear layers of size 300 and 100.
    """
    def __init__(self):
        super().__init__()
        self.l1 = BayesLinear(28*28, 300)
        self.l2 = BayesLinear(300, 100)
        self.l3 = BayesLinear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


if __name__ == "__main__":
    main()
