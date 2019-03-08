from torch import nn
from models.layers import Flatten


class LeNet300_100(nn.Module):
    def __init__(self, activation='Tanh'):
        super().__init__()

        if activation == 'Tanh':
            activation = nn.Tanh
        elif activation == 'ReLU':
            activation = nn.ReLU
        else:
            raise TypeError('Activation should be either Tanh or ReLU.')

        self.classificator = nn.Sequential(
            Flatten(),
            nn.Linear(28*28, 300),
            activation(),
            nn.Linear(300, 100),
            activation(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        return self.classificator.forward(x)
