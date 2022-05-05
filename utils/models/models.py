import os.path

import numpy as np

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


def get_optimizer(model):
    # instantiate the optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=1)

    # instantiate the learning decay scheduler
    lr_decay = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
    return optimizer, lr_decay


def train(n_epochs: int, loaders: dict, model, optimizer, criterion, use_cuda: bool,
          save_path: str, learning_decay_scheduler):
    """
    This function trains the model and shows the progress.

    Parameters:
        n_epochs (int): Number of epochs to train for
        loaders (dict): Dictionary of loaders to use
        model: Model being used
        optimizer: Selected optimizer
        criterion: Loss function
        use_cuda (bool): If GPU is enables or not
        save_path (str): Path to store the results in
        learning_decay_scheduler: Learning rate decay scheduler to use

    Returns:
        A trained model
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # record the average training loss, using something like
            optimizer.zero_grad()

            # get the final outputs
            output = model(data)

            # calculate the loss
            loss = criterion(output, target)

            # start back propagation
            loss.backward()

            # update the weights
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['val']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # update average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss /= len(loaders['train'].sampler)
        valid_loss /= len(loaders['val'].sampler)

        # print training/validation statistics every 5 epochs
        if epoch % 5 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

        # if the validation loss has decreased, save the model at the filepath
        # stored in save_path
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))

            if not os.path.exists('checkpoints/'):
                os.mkdir(save_path)

            torch.save(model.state_dict(), 'checkpoints/' + save_path)
            valid_loss_min = valid_loss

        # update learning rate decay
        learning_decay_scheduler.step()

    return model


def test(loaders, model, criterion, use_cuda):
    """
    This functions calculates the correctness and shows the results of the architecture.

    Parameters:
        loaders: Dictionary that stores all three loaders
        model: Model used for implementation
        criterion: Loss function
        use_cuda: If GPU is available or not

    Returns:
        The accuracy of the model
    """
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    # show the accuracy
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))