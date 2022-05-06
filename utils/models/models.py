import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet34, resnet18, vgg16

import os
import numpy as np
from utils.utils import Dictionary
from utils.visualize.visualize import suggest_locations, pytorch_results
from utils.models.hyper_params import get_optim
from utils.preprocess.preprocess import pytorch_loaders

# create criterion
criterion = nn.CrossEntropyLoss()


def change_ending(model, name):
    """
    Change the FC layer (resnet18, resnet34) or last classifier layer (vgg16)
    to match the number of classes.

    Parameters
    ----------
    model : torchvision.models.resnet.ResNet
        CNN model (options: vgg16, resnet18, resnet34)

    name :  str
        Name of the CNN model (options: vgg16, resnet18, resnet34)

    Returns
    -------
    CNN model with changed last layer parameters

    """
    if name == 'vgg16':
        # params for vgg16
        for param in model.features.parameters():
            param.requires_grad = False

        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 50)

    elif name == 'resnet34' or name == 'resnet18':
        model.fc = nn.Linear(in_features=512, out_features=50)

    return model


def train(path='data', model_name='resnet34', mode='pytorch',
          lr=0.01, batch_size=16, epochs=50, optimizer='adam'):
    """
    Trains the model in pure PyTorch or FastAI.

    Parameters
    ----------
    path       : str (default='data')
        Path to train/test or train/val/test location
    model_name : str (default="resnet34")
        Name of the CNN model (options: vgg16, resnet18, resnet34)
    mode       : str (default="pytorch")
        Name of the library to train the model in (options: pytorch, fastai)
    lr         : int (default=0.01)
        Learning rate


    mode : str
        param lr: (Default value = 'pytorch')
    batch_size :
        param epochs: (Default value = 16)
    optimizer :
        return: (Default value = 'adam')
    model_name :
         (Default value = 'resnet34')
    lr :
         (Default value = 0.01)
    epochs :
         (Default value = 50)

    Returns
    -------

    """
    cuda = torch.cuda.is_available()

    if mode == 'pytorch':
        model = get_model(model_name, trained=False)
        loaders, dictionary = pytorch_loaders(batch_size, path, cuda)
        optimizer, lr_decay = get_optim(model, optimizer, lr=lr,
                                        learning_decay=True, nesterov=True)
        # train model
        model, train_loss, val_loss = train_pytorch(epochs, loaders, model, optimizer,
                                                    cuda, lr_decay, model_name=model_name)
        # visualize the results
        pytorch_results(train_loss, val_loss)

        # test the model
        confused_with, test_dict, test_loss = test_pytorch(loaders, model, criterion, cuda, dictionary)

        # show final results
        print(f'''========== Ending Training ==========
        Train loss: {train_loss[-1]}
        Valid loss: {val_loss[-1]}
        Test  loss: {test_loss}
        ''')


def train_pytorch(epochs, loaders, model, optimizer, cuda,
                  lr_decay, save_path='checkpoints', model_name='resnet34'):
    """

    Parameters
    ----------
    epochs :
        param loaders:
    model :
        param optimizer:
    cuda :
        param lr_decay:
    save_path :
        param model_name: (Default value = 'checkpoints')
    loaders :
        
    optimizer :
        
    lr_decay :
        
    model_name :
         (Default value = 'resnet34')

    Returns
    -------

    """
    # create empty lists to store values
    train_losses = []
    valid_losses = []

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, epochs + 1):
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
            if cuda:
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

            # update training loss
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['val']):
            # move to GPU
            if cuda:
                data, target = data.cuda(), target.cuda()

            # update average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        # update training and validation losses
        train_loss /= len(loaders['train'].sampler)
        valid_loss /= len(loaders['val'].sampler)

        # append loss results
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics every 5 epochs
        if epoch % 5 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

        # if the validation loss has decreased, save the model at the filepath stored in save_path
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), os.path.join(save_path, model_name + '.pt'))
            valid_loss_min = valid_loss

        # update learning rate decay
        if lr_decay:
            lr_decay.step()

    return model, train_losses, valid_losses


def test_pytorch(loaders, model, criterion, cuda, dictionary):
    """
    Parameters
    ----------
    loaders :
        param model:
    criterion :
        param cuda:
    dictionary :
        return:
    model :
        
    cuda :
        

    Returns
    -------

    """
    test_loss, correct, total = 0., 0., 0.

    # keep track of correctly classified
    class_correct = np.zeros(50)
    class_total = np.zeros(50)

    # store correct and missed predictions
    test_dict, confused_with = {}, {}

    # set model to eval mode
    model.eval()

    # start testing
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if cuda:
            data, target = data.cuda(), target.cuda()

        # forward pass
        output = model(data)

        # calculate loss
        loss = criterion(output, target)

        # update averate test loss
        test_loss += (1 / (batch_idx + 1)) * (loss.data.item() - test_loss)

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true classes
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        for i in range(len(target)):
            label = target.data[i]

            if int(label) not in confused_with:
                confused_with[int(label)] = []
            else:
                confused_with[int(label)].append(pred[i].item())

            class_correct[label] += np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy()[i].item()
            class_total[label] += 1

    # show loss and accuracy
    print('Test Loss: {:.6f}\n'.format(test_loss))
    print('\nTest accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    # get final predictions
    for i in range(50):
        if class_total[i] > 0:
            name = dictionary.get_item(i)
            accuracy = 100 * class_correct[i] / class_total[i]
            test_dict[name] = accuracy

    return confused_with, test_dict, test_loss


def get_model(model_name, trained=True):
    """

    Parameters
    ----------
    model_name :
        param trained:
    trained :
         (Default value = True)

    Returns
    -------

    """
    # get cuda condition
    cuda = torch.cuda.is_available()

    model = None

    if model_name.lower() == 'resnet34':
        model = change_ending(resnet34(pretrained=True), 'resnet34')
        if trained:
            model.load_state_dict(torch.load('checkpoints/resnet34.pt'))
    elif model_name.lower() == 'resnet18':
        model = change_ending(resnet18(pretrained=True), 'resnet18')
        if trained:
            model.load_state_dict(torch.load('checkpoints/resnet18.pt'))
    elif model_name.lower() == 'vgg16':
        model = change_ending(vgg16(pretrained=True), 'vgg16')
        if trained:
            model.load_state_dict(torch.load('checkpoints/vgg16.pt'))

    # set model to cuda if available
    if cuda:
        model = model.cuda()

    return model


def predict_image(img, title, model_name='resnet34'):
    """

    Parameters
    ----------
    img :
        param title:
    model_name :
        return: (Default value = 'resnet34')
    title :
        

    Returns
    -------

    """

    # open checkpoints and predict the image
    model = get_model(model_name, trained=True)

    # create dictionary
    dictionary = Dictionary(datasets.ImageFolder('output/train'))

    # show image and TOP-3 predictions
    suggest_locations(img, model, dictionary, name=title, k=3)
