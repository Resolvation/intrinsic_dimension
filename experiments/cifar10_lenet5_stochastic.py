import torch
from torch import optim
from tensorboardX import SummaryWriter

from core.data import cifar10_loaders
from core.train import train, test_classifier
from core.utils import SGVLB
from models.LeNet5_stochastic import LeNet5


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda')
writer = SummaryWriter('../writers/CIFAR10_LeNet5_stochastic')

train_loader, test_loader = cifar10_loaders()
model = LeNet5().cuda()

trainset_criterion = SGVLB(model, len(train_loader.dataset))
testset_criterion = SGVLB(model, len(test_loader.dataset))

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    writer.add_scalar('training/learning_rate', lr, epoch)

    avg_loss, accuracy = train(
        model, device, train_loader, trainset_criterion,
        optimizer, epoch, True)
    writer.add_scalar('training/loss', avg_loss, epoch)
    writer.add_scalar('training/accuracy', accuracy, epoch)

    if epoch % 10 == 0:
        print('Mean:')
        print(' Train set:')
        avg_loss, accuracy = test_classifier(
            model, device, train_loader, trainset_criterion, True)
        writer.add_scalar('testing/det/trainset/loss', avg_loss, epoch)
        writer.add_scalar('testing/det/trainset/accuracy', accuracy, epoch)

        print(' Test set:')
        avg_loss, accuracy = test_classifier(
            model, device, test_loader, testset_criterion, True)
        writer.add_scalar('testing/det/testset/loss', avg_loss, epoch)
        writer.add_scalar('testing/det/testset/accuracy', accuracy, epoch)

        print('Ensemble:')
        print(' Train set:')
        avg_loss, accuracy = test_classifier(
            model, device, train_loader, trainset_criterion, True, 10)
        writer.add_scalar('testing/ens/trainset/loss', avg_loss, epoch)
        writer.add_scalar('testing/ens/trainset/accuracy', accuracy, epoch)

        print(' Test set:')
        avg_loss, accuracy = test_classifier(
            model, device, test_loader, testset_criterion, True, 10)
        writer.add_scalar('testing/ens/testset/loss', avg_loss, epoch)
        writer.add_scalar('testing/ens/testset/accuracy', accuracy, epoch)

        for i, child in enumerate(model.children()):
            writer.add_histogram(f'std/{i+1}',
                                 (child.log_sigma_sqr.view(-1) / 2).exp(),
                                 epoch)

torch.save(model.state_dict(), '../tars/CIFAR10_LeNet5_stochastic_'
                               f'{accuracy:.02f}.tar')
writer.close()
