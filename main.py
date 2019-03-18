import argparse

import torch
from torch import nn, optim
from tensorboardX import SummaryWriter

from core.data import cifar10_loaders, mnist_loaders
from core.train import eval_determenistic, eval_stochastic, train
from core.utils import adjust_learning_rate, linear_lr, SGVLB
from models import LeNet300_100, LeNet300_100_stochastic
from models import LeNet5, LeNet5_stochastic, LeNet5_stochastic_id


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='CIFAR10')
parser.add_argument('-m', '--model', default='LeNet5')
parser.add_argument('-p', '--pretrained', default=False)
parser.add_argument('-s', '--seed', default=42, type=int)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-id', '--intrinsic_dimension', default=10, type=int)
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device('cuda')

if args.dataset == 'MNIST':
    n_classes = 10
    train_loader, test_loader = mnist_loaders()
elif args.dataset == 'CIFAR10':
    n_classes = 10
    train_loader, test_loader = cifar10_loaders()
else:
    raise parser.error('Wrong dataset name.')

if args.model == 'LeNet300_100':
    model = LeNet300_100(n_classes).cuda()
elif args.model == 'LeNet300_100_stochastic':
    model = LeNet300_100_stochastic(n_classes).cuda()
elif args.model == 'LeNet5':
    model = LeNet5(n_classes).cuda()
elif args.model == 'LeNet5_stochastic':
    model = LeNet5_stochastic(n_classes).cuda()
elif args.model == 'LeNet5_stochastic_id':
    model = LeNet5_stochastic_id(args.intrinsic_dimension, n_classes).cuda()
else:
    raise parser.error('Wrong model name.')

if args.pretrained:
    model.load_weights('tars/'+args.pretrained)

if 'stochastic' not in args.model:
    criterion = nn.CrossEntropyLoss()
else:
    criterion = SGVLB(model, len(train_loader.dataset))
    testset_criterion = SGVLB(model, len(test_loader.dataset))

name = 'pretrained_' if args.pretrained else ''
name += args.dataset + '_' + args.model
writer = SummaryWriter(f'writers/{name}')

lr = init_lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=lr)

if args.pretrained:
    n_epochs = 100
    alpha = 0
else:
    n_epochs = 400
    alpha = 1

if 'stochastic' not in args.model:
    eval_determenistic(writer, model, device, train_loader,
                       test_loader, criterion, 0, args.verbose)
else:
    eval_stochastic(writer, model, device, train_loader, test_loader,
                    criterion, testset_criterion, 0, args.verbose)
    model.log_weights(writer, 0)

for epoch in range(1, n_epochs + 1):
    lr = linear_lr(epoch, n_epochs, alpha) * init_lr
    adjust_learning_rate(optimizer, lr)
    writer.add_scalar('training/learning_rate', lr, epoch)

    res = train(model, device, train_loader, criterion,
                optimizer, epoch, args.verbose)
    writer.add_scalar('training/loss', res[0], epoch)
    writer.add_scalar('training/nllloss', res[1], epoch)
    writer.add_scalar('training/eceloss', res[2], epoch)
    writer.add_scalar('training/mceloss', res[3], epoch)
    writer.add_scalar('training/accuracy', res[4], epoch)

    if epoch % 10 == 0:
        if 'stochastic' not in args.model:
            eval_determenistic(writer, model, device, train_loader,
                               test_loader, criterion, epoch, args.verbose)
        else:
            eval_stochastic(writer, model, device, train_loader, test_loader,
                            criterion, testset_criterion, epoch, args.verbose)
            model.log_weights(writer, epoch)


torch.save(model.state_dict(), f'tars/{name}_{res[4]:.02f}.tar')
writer.close()
