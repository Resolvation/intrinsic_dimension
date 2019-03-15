import torch
from torch import optim
from tensorboardX import SummaryWriter

from core.data import cifar10_loaders
from core.train import train, eval_stochastic
from core.utils import adjust_learning_rate, linear_lr, SGVLB
from models.LeNet5_stochastic import LeNet5_stochastic


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda')
writer = SummaryWriter('../writers/CIFAR10_LeNet5_stochastic')

train_loader, test_loader = cifar10_loaders()
model = LeNet5_stochastic().cuda()

trainset_criterion = SGVLB(model, len(train_loader.dataset))
testset_criterion = SGVLB(model, len(test_loader.dataset))

lr = lr_init = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

eval_stochastic(writer, model, device, train_loader, test_loader,
                trainset_criterion, testset_criterion, 0)
model.log_weights(writer, 0)

n_epochs = 400
for epoch in range(1, n_epochs + 1):
    lr = linear_lr(epoch, n_epochs) * lr_init
    adjust_learning_rate(optimizer, lr)
    writer.add_scalar('training/learning_rate', lr, epoch)

    avg_loss, accuracy = train(
        model, device, train_loader, trainset_criterion,
        optimizer, epoch, True)
    writer.add_scalar('training/loss', avg_loss, epoch)
    writer.add_scalar('training/accuracy', accuracy, epoch)

    if epoch % 10 == 0:
        eval_stochastic(writer, model, device, train_loader, test_loader,
                        trainset_criterion, testset_criterion, epoch)
        model.log_weights(writer, epoch)

torch.save(model.state_dict(), '../tars/CIFAR10_LeNet5_stochastic_'
                               f'{accuracy:.02f}.tar')
writer.close()
