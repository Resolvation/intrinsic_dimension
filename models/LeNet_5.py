from torch import nn
from torch.nn import functional as F
from torch.nn import init


class LeNet_5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2)
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def init_weights(self):
        init.kaiming_normal_(self.conv1.weight, mode='fan_out')
        init.kaiming_normal_(self.conv2.weight, mode='fan_out')
        init.kaiming_normal_(self.fc1.weight, mode='fan_out')
        init.constant_(self.fc1.bias, 0)
        init.kaiming_normal_(self.fc2.weight, mode='fan_out')
        init.constant_(self.fc2.bias, 0)
        init.kaiming_normal_(self.fc3.weight, mode='fan_out')
        init.constant_(self.fc3.bias, 0)
        return self
