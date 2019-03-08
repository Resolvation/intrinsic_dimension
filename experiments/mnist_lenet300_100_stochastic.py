import torch
from torch import optim
from tensorboardX import SummaryWriter

from core.data import mnist_loaders
from core.train import train, test_classifier
from core.utils import SGVLB
from models.LeNet300_100_stochastic import LeNet300_100


torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device('cuda')
writer = SummaryWriter('../writers/MNIST_LeNet300_100_stochastic')

train_loader, test_loader = mnist_loaders()
model = LeNet300_100().cuda()

trainset_criterion = SGVLB(model, len(train_loader.dataset))
testset_criterion = SGVLB(model, len(test_loader.dataset))

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

n_epochs = 20
for epoch in range(1, n_epochs + 1):
    writer.add_scalar('training/learning_rate', lr, epoch)

    avg_loss, accuracy = train(
        model, device, train_loader, trainset_criterion,
        optimizer, epoch, True)
    writer.add_scalar('training/loss', avg_loss, epoch)
    writer.add_scalar('training/accuracy', accuracy, epoch)

    if epoch % 10 == 0:
        print('Train set:')
        avg_loss, accuracy = test_classifier(
            model, device, train_loader, trainset_criterion, True)
        writer.add_scalar('testing/trainset/loss', avg_loss, epoch)
        writer.add_scalar('testing/trainset/accuracy', accuracy, epoch)

        print('Test set:')
        avg_loss, accuracy = test_classifier(
            model, device, test_loader, testset_criterion, True)
        writer.add_scalar('testing/testset/loss', avg_loss, epoch)
        writer.add_scalar('testing/testset/accuracy', accuracy, epoch)

torch.save(model.state_dict(), '../tars/MNIST_LeNet300_100_stochastic'
                               f'{accuracy:.02f}.tar')
writer.close()
