import torch
from torch import nn
from torchvision import datasets
from torchvision.models import resnet34, resnet18, vgg16

import os
import numpy as np
import pandas as pd
from utils.utils import Dictionary
from utils.visualize.visualize import suggest_locations, pytorch_results
from utils.models.hyper_params import get_optim
from utils.preprocess.preprocess import pytorch_loaders

# import fastai library
import fastbook
from fastbook import *
import fastai

# create criterion
criterion = nn.CrossEntropyLoss()


def change_ending(model, name):
    """
    Change the FC layer (resnet18, resnet34) or last classifier layer (vgg16)
    to match the number of classes.

    Parameters
    ----------
    model : torchvision.models object
        CNN model (options: vgg16, resnet18, resnet34)

    name :  str
        Name of the CNN model (options: vgg16, resnet18, resnet34)

    Returns
    -------
    CNN model with changed last layer parameters

    """
    # change the last layer
    if name == 'vgg16':
        # params for vgg16
        for param in model.features.parameters():
            param.requires_grad = False

        # set the new nn layer
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 50)

    elif name == 'resnet34' or name == 'resnet18':
        # set the last fc layer to 50
        model.fc = nn.Linear(in_features=512, out_features=50)

    return model


def train(path='data', model_name='resnet34', mode='pytorch', lr=0.01,
          batch_size=16, epochs=50, optimizer='adam', save_name=None):
    """
    Trains the model in pure PyTorch or FastAI. Shows the train/validation
    losses in a plot and prints train/val/test losses at the end of the training.

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
    batch_size : int (default=16)
        Batch size
    epochs     : int (default=50)
        Number of epochs
    optimizer  : str (default="adam")
        Optimizer to use (options: adam, adagrad, sgd)
    save_name  : str (default=None)
        Name of the checkpoint
    """

    # get cuda parameters
    cuda = torch.cuda.is_available()

    # train a pytorch model
    if mode == 'pytorch':
        # get model
        model = get_model(model_name, trained=False)
        # get loaders and dictionary that holds cleaned labels
        loaders, dictionary = pytorch_loaders(batch_size, path, cuda)
        # get optimizers
        optimizer, lr_decay = get_optim(model, optimizer, lr=lr,
                                        learning_decay=True, nesterov=True)
        # train model
        model, train_loss, val_loss = train_pytorch(epochs, loaders, model, optimizer,
                                                    cuda, lr_decay, model_name=model_name,
                                                    save_name=save_name)
        # visualize the results
        pytorch_results(train_loss, val_loss)

        # test the model
        confused_with, test_dict, test_loss = test_pytorch(loaders, model, cuda, dictionary)

        # show final results
        print(f'''========== Ending Training ==========
        Train loss: {train_loss[-1]}
        Valid loss: {val_loss[-1]}
        Test  loss: {test_loss}
        ''')
    elif mode == 'fastai':
        dls, model = train_fastai(path='output', bs=batch_size, img_size=226, shuffle=True,
                                  freeze=3, epochs=epochs, model=model_name)


def train_pytorch(epochs, loaders, model, optimizer, cuda, lr_decay,
                  save_path='checkpoints', model_name='resnet34', save_name=None):
    """
    Train a PyTorch model of choice (vgg16, resnet18, resnet34).

    Parameters
    ----------
    epochs      : int
        Number of epochs to train for
    loaders     : dict
        Train/val/test loaders stored in a dictionary
    model       : torchvision.models object
        CNN model (options: vgg16, resnet18, resnet34)
    cuda        : bool
        Whether CUDA is enabled
    lr_decay    : torch.optim.ExponentialLR
        Learning decay scheduler
    save_path   : str (default="checkpoints")
        Path to store the best model in
    model_name  : str (default="resnet34")
        Name of the model (vgg16, resnet18, resnet34)
    save_name   : str (default=None)
        Name of the checkpoint

    Returns
    -------
    Trained model and lists of train/validation losses
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
            path = model_name + '.pt' if save_name is None else save_name + '.pt'
            torch.save(model.state_dict(), os.path.join(save_path, path))
            valid_loss_min = valid_loss

        # update learning rate decay
        if lr_decay:
            lr_decay.step()

    return model, train_losses, valid_losses


def test_pytorch(loaders, model, cuda, dictionary):
    """
    Test the model's performance

    Parameters
    ----------
    loaders     : dict
        Train/val/test loaders stored in a dictionary
    model       : torchvision.models object
        CNN model (options: vgg16, resnet18, resnet34)
    cuda        : bool
        Whether CUDA is enabled
    dictionary  : object
        Dictionary object that stores classes

    Returns
    -------
    `confused_with` - dictionary that stores all classes that were wrongly predicted
                      in the descending order
    `test_dict`     - dictionary that stores all classes and test prediction information
    `test_loss`     - list that stores loss metrics
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
    Create model and load from checkpoint if `trained` is enabled

    Parameters
    ----------
    model_name  : str
        Name of the CNN model (vgg16, resnet18, resnet34)
    trained     : bool (default=True)
        Whether to train a new model or load existing

    Returns
    -------
    Trained model loaded from checkpoints (if trained=True) or
    loads a new pre-trained model from torchvision.models

    """
    # get cuda condition
    cuda = torch.cuda.is_available()

    # create a placeholder
    model = None

    # load a new or fully-trained model
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


# create a dataloader
def get_data_loader(path='output', bs=16, img_size=226, shuffle=True):
    datablock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                          get_items=partial(get_image_files, folders=['train', 'val']),
                          splitter=GrandparentSplitter(train_name='train', valid_name='val'),
                          get_y=parent_label,
                          item_tfms=[Resize((img_size, img_size), method='pad', pad_mode='zeros'),
                                     ToTensor()],
                          batch_tfms=[*aug_transforms(flip_vert=True, max_rotate=15,
                                                      pad_mode='zeros', max_lighting=0.6),
                                      Normalize.from_stats(*imagenet_stats)])
    return datablock.dataloaders(path, bs=bs, drop_last=True, shuffle=shuffle)


def train_fastai(path='output', bs=16, img_size=226, shuffle=True,
                freeze=3, epochs=20, model=resnet34):
    """
  Train and save best model.
    :param path:     Path to a folder (has to have train/val folders)
    :param bs:       Batch size, default=16
    :param img_size: Image size, default: 256x256
    :param shuffle:  Whether to shuffle data, default: true
    :param freeze:   Number of epochs to train for with frozen layers, default=3
    :param epochs:   Number of epochs to train the whole model for, default=10
    :param model:    Model to train with, default=ResNet50

  :return: DataLoader (dls) and trained model (learn)
  """
    model_name = str(model).split(' ')[1]

    print(f"""
  ===== Start training =====
  Epochs: {epochs}
  Model: {model_name}
  Freeze epochs: {freeze}
  Batch size: {bs}
  """)

    # get the dataloader
    dls = get_data_loader(path, bs, img_size, shuffle)
    # create a model, save with the highest accuracy and F1 score
    learn = cnn_learner(dls, model, metrics=[accuracy, F1Score(average='macro')],
                        cbs=SaveModelCallback(monitor='f1_score', comp=np.greater))
    # freeze and train
    learn.fine_tune(epochs, freeze_epochs=freeze,
                    cbs=[ShowGraphCallback(), TrackerCallback()])
    # plot confusion matrix and classification report
    plot_results(learn)

    # save checkpoint
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')
    # get the unique model name
    acc = round(learn.final_record[1], 2)
    name = model_name + '_' + str(acc) + 'acc'
    learn.save(os.path.join('checkpoints', name))

    # return DataLoader and trained model
    return dls, learn


def plot_results(learn):
    """
  This function creates a confusion matrix and classification report.
    :param learn: FastAI's cnn_learner instance
  """
    interp = ClassificationInterpretation.from_learner(learn)
    interp.print_classification_report()

    # print the "most confused" classes
    print('===============')
    print('Most confused')
    print(pd.DataFrame({'confused': interp.most_confused(min_val=2)}))
    print('===============')


def predict_image(img, title, model_name='resnet34'):
    """
    Predict the given image and show TOP-3 predictions
    histogram side-by-side

    Parameters
    ----------
    img         : PIL.Image
        PIL.Image
    title       : str
        Folder location of the image
    model_name  : str (default="resnet34")
        Name of the model (options: vgg16, resnet18, resnet34)
    """

    # open checkpoints and predict the image
    model = get_model(model_name, trained=True)

    # create dictionary
    dictionary = Dictionary(datasets.ImageFolder('output/train'))

    # show image and TOP-3 predictions
    suggest_locations(img, model, dictionary, name=title, k=3)
