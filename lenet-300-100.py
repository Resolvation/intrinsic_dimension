import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.data_loader import mnist_data
from core.evaluate import evaluate
from core.train import train
from core.utils import save


def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet_300_100().to(device)
    train_loader, test_loader = mnist_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(device, model, train_loader, test_loader, criterion, optimizer)

    save(model, "LeNet_300_100")


class LeNet_300_100(nn.Module):
    """
    Neural modelwork with two linear layers of size 300 and 100.
    """
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28*28, 300)
        self.l2 = nn.Linear(300, 100)
        self.l3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


if __name__ == "__main__":
    main()
