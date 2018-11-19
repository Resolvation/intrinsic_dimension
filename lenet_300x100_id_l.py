import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from core.data_loader import mnist_data
from core.layers import LinearOffset
from core.train import train
from core.utils import save


def main():
    torch.manual_seed(42)

    writer = SummaryWriter("./logs/LeNet_300_100_ID_L")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet_300_100_ID_L(450, 250, 50).to(device)

    train_loader, test_loader = mnist_data()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    train(writer, device, model, train_loader, test_loader,
          criterion, optimizer)

    save(model, "LeNet_300_100_ID_L")
    writer.close()


class LeNet_300_100_ID_L(nn.Module):
    """
    Neural modelwork with two linear layers of size 300 and 100 in intristic
    dimentions of size d1, d2, d3.
    """
    def __init__(self, d1, d2, d3):
        super().__init__()
        self.l1 = LinearOffset(d1, 28*28, 300)
        self.l2 = LinearOffset(d2, 300, 100)
        self.l3 = LinearOffset(d3, 100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


if __name__ == "__main__":
    main()
