import torch
from torch import optim
from tensorboardX import SummaryWriter

from core.data import cifar10_loaders
from core.train import train, eval_stochastic
from core.utils import SGVLB
from models.LeNet5_stochastic_id import LeNet5_stochastic_id


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = torch.device('cuda')
writer = SummaryWriter('../writers/pretrained_CIFAR10_LeNet5_stochastic_id')

train_loader, test_loader = cifar10_loaders()
model = LeNet5_stochastic_id(50)
model.load_weights('../tars/CIFAR10_LeNet5_baseline_73.97.tar').cuda()

trainset_criterion = SGVLB(model, len(train_loader.dataset))
testset_criterion = SGVLB(model, len(test_loader.dataset))

lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

eval_stochastic(writer, model, device, train_loader, test_loader,
                trainset_criterion, testset_criterion, 0)
writer.add_histogram(f'std/', (model.log_sigma_sqr.view(-1) / 2).exp(), 0)

n_epochs = 100
for epoch in range(1, n_epochs + 1):
    writer.add_scalar('training/learning_rate', lr, epoch)

    avg_loss, accuracy = train(
        model, device, train_loader, trainset_criterion,
        optimizer, epoch, True)
    writer.add_scalar('training/loss', avg_loss, epoch)
    writer.add_scalar('training/accuracy', accuracy, epoch)

    if epoch % 10 == 0:
        eval_stochastic(writer, model, device, train_loader, test_loader,
                        trainset_criterion, testset_criterion, epoch)
        writer.add_histogram(f'std/', (model.log_sigma_sqr.view(-1) / 2).exp(),
                             epoch)



torch.save(model.state_dict(), '../tars/pretrained_CIFAR10_LeNet5_stochastic_'
                               f'id_{accuracy:.02f}.tar')
writer.close()
