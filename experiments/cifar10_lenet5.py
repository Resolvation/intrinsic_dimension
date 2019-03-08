import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from core.data import cifar10_loaders
from core.train import train, test_classifier
from core.utils import linear_lr, adjust_learning_rate
from models.LeNet5 import LeNet5


torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device('cuda')
writer = SummaryWriter('../writers/CIFAR10_LeNet5_linear_uniform')

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
        print('Train set:')
        avg_loss, accuracy = test_classifier(
            model, device, train_loader, criterion, True)
        writer.add_scalar('testing/trainset/loss', avg_loss, epoch)
        writer.add_scalar('testing/trainset/accuracy', accuracy, epoch)

        print('Test set:')
        avg_loss, accuracy = test_classifier(
            model, device, test_loader, criterion, True)
        writer.add_scalar('testing/testset/loss', avg_loss, epoch)
        writer.add_scalar('testing/testset/accuracy', accuracy, epoch)

torch.save(model.state_dict(), '../tars/CIFAR10_LeNet5_linear_uniform'
                               f'{accuracy:.02f}.tar')
writer.close()
