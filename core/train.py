import datetime as dt

import torch.nn.functional as F

import core.utils as utils


def stdn_loss(net, dataset_len):
    """
    Return loss function for variational nn with stdn prior.
    """
    def loss(outputs, labels):
        return F.cross_entropy(outputs, labels) + net.KL()/dataset_len
    return loss


def train(device, net, train_loader, criterion, optimizer, n_epochs=200):
    """
    Train net on device using optimizer to minimize criterion on train_loader
    for n_epochs.
    """
    print("Training:")
    steps, losses = [0], []
    init_lr = utils.lr(optimizer)
    start = dt.datetime.now()
    for epoch in range(n_epochs):
        epoch_start = dt.datetime.now()
        optimal_lr = utils.lr_linear(epoch, n_epochs/2, n_epochs, init_lr)
        utils.adjust_learning_rate(optimizer, optimal_lr)
        batch_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            if (batch_idx + 1) % n_epochs == 0:
                steps.append(steps[-1]+labels.shape[0])
                losses.append(loss.item())
        if (epoch + 1) % 5 == 0:
            print("Epoch: {:>03}    Loss: {:>010.4f}    Time: {:8}".format(epoch+1, batch_loss/len(train_loader),
                                                                 str(dt.datetime.now()-epoch_start).split('.')[0]))
    print("Time: {:8}".format(str(dt.datetime.now()-start).split('.')[0]))
