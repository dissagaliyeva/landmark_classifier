import torch
from torch import nn
from torch import optim


def get_optim(model, optim_name, lr=0.01,
              learning_decay=False, nesterov=True):
    """

    Parameters
    ----------
    model :
        param optim_name:
    lr :
        param learning_decay: (Default value = 0.01)
    nesterov :
        return: (Default value = True)
    optim_name :
        
    learning_decay :
         (Default value = False)

    Returns
    -------

    """
    optimizer = None
    lr_decay  = None
    momentum  = 0.9

    optim_name = optim_name.lower()
    assert optim_name in ['sgd', 'adagrad', 'adam'], f'{optim_name} optimizer is not on the list.' \
                                                     f'Please choose one of the following: sgd, adagrad, adam'

    if optim_name == 'sgd':
        optimizer = optim.SGD([
            {'params': model.layer1.parameters(), 'lr': lr / 10, 'momentum': momentum, 'nesterov':True},
            {'params': model.layer2.parameters(), 'lr': lr / 10, 'momentum': momentum, 'nesterov': True},
            {'params': model.layer3.parameters(), 'lr': lr / 10, 'momentum': momentum, 'nesterov': True},
            {'params': model.layer4.parameters(), 'lr': lr / 10, 'momentum': momentum, 'nesterov': True},
            {'params': model.fc.parameters(), 'lr': lr, 'momentum': momentum, 'nesterov': nesterov}],
            lr=lr, momentum=momentum, nesterov=True)
    elif optim_name == 'adagrad':
        optimizer = optim.Adagrad([
            {'params': model.layer1.parameters(), 'lr': lr / 10},
            {'params': model.layer2.parameters(), 'lr': lr / 10},
            {'params': model.layer3.parameters(), 'lr': lr / 10},
            {'params': model.layer4.parameters(), 'lr': lr / 10},
            {'params': model.fc.parameters(), 'lr': lr}],
            lr=lr)
    elif optim_name == 'adam':
        optimizer = optim.Adam([
            {'params': model.layer1.parameters(), 'lr': lr / 10, 'amsgrad': True},
            {'params': model.layer2.parameters(), 'lr': lr / 10, 'amsgrad': True},
            {'params': model.layer3.parameters(), 'lr': lr / 10, 'amsgrad': True},
            {'params': model.layer4.parameters(), 'lr': lr / 10, 'amsgrad': True},
            {'params': model.fc.parameters(), 'lr': lr, 'amsgrad': True}],
            lr=lr, amsgrad=True)

    if learning_decay:
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    return optimizer, lr_decay