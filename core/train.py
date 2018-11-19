import datetime as dt

import torch.nn.functional as F
from tensorboardX import SummaryWriter

from core.evaluate import evaluate
import core.utils as utils


def stdn_loss(model, dataset_len):
    """
    Return loss function for variational nn with stdn prior.
    """
    def loss(outputs, labels):
        return F.cross_entropy(outputs, labels) + model.KL()/dataset_len
    return loss


def train(writer, device, model, train_loader, test_loader,
          criterion, optimizer, test_criterion=None, n_epochs=400):
    """
    Train model on device using optimizer to minimize criterion on train_loader
    for n_epochs.
    """
    print("Training:")
    init_lr = utils.lr(optimizer)
    start = dt.datetime.now()
    for epoch in range(1, n_epochs+1):
        epoch_start = dt.datetime.now()
        epoch_correct, epoch_loss = 0., 0.
        optimal_lr = utils.lr_linear(epoch, n_epochs/2, n_epochs, init_lr)
        writer.add_scalar("training/learning rate", optimal_lr, epoch)
        utils.adjust_learning_rate(optimizer, optimal_lr)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pred = outputs.max(1, keepdim=True)[1]
            epoch_correct += pred.eq(labels.view_as(pred)).sum().item()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        epoch_accuracy = 100 * epoch_correct / len(train_loader.dataset)
        print("Epoch: {:>03}    Loss: {:>010.4f}    Accuracy: {:>.2f}    "
              "Time: {:8}".format(epoch, epoch_loss, epoch_accuracy,
                         str(dt.datetime.now()-epoch_start).split('.')[0]))
        writer.add_scalar("training/average loss", epoch_loss, epoch)
        writer.add_scalar("training/accuracy", epoch_accuracy, epoch)
        if epoch % 20 == 0:
            print("After {:} epochs".format(epoch))
            logger = {"writer":writer, "tag":"testing/trainset/", "global_step":epoch}
            print("Trainset:")
            evaluate(logger, device, model, train_loader,
                     criterion)
            print("Testset:")
            logger["tag"] = "testing/testset/"
            if test_criterion is None:
                evaluate(logger, device, model, test_loader,
                         criterion)
            else:
                evaluate(logger, device, model, test_loader,
                         test_criterion)
    print("Time: {:8}".format(str(dt.datetime.now()-start).split('.')[0]))
