import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from core.data import cifar10_loaders
from core.train import train, eval_determenistic
from core.utils import linear_lr, adjust_learning_rate
from models.LeNet5 import LeNet5


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda')
writer = SummaryWriter('../writers/CIFAR10_LeNet5_baseline')

criterion = nn.CrossEntropyLoss()
train_loader, test_loader = cifar10_loaders()

model = LeNet5().cuda()

lr = init_lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

n_epochs = 400
for epoch in range(1, n_epochs + 1):
    lr = linear_lr(epoch, n_epochs) * init_lr
    adjust_learning_rate(optimizer, lr)
    writer.add_scalar('training/learning_rate', lr, epoch)

    avg_loss, accuracy = train(
        model, device, train_loader, criterion, optimizer, epoch, True)
    writer.add_scalar('training/loss', avg_loss, epoch)
    writer.add_scalar('training/accuracy', accuracy, epoch)

    if epoch % 10 == 0:
        eval_determenistic(writer, model, device, train_loader,
                           test_loader, criterion, epoch, True)

torch.save(model.state_dict(), '../tars/CIFAR10_LeNet5_baseline_'
                               f'{accuracy:.02f}.tar')
writer.close()
