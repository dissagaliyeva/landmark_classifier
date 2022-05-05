import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F

import re
import os
import random
import splitfolders

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_transforms(img_size, crop_size):
    # define transformations
    return {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
            transforms.RandomRotation((-30, 30)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD
            )
        ]),

        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD
            )
        ])
    }


def pytorch_loaders(n_batch, path='data', batch=True,
                    cuda=True, img_size=226, crop_size=224) -> tuple:
    """


    :param crop_size:
    :param img_size:
    :param n_batch:
    :param path:
    :param batch:
    :param cuda:
    :return:
    """

    transform = get_transforms(img_size, crop_size)

    # process single image
    if not batch:
        # get random image
        img = get_random_image(path + '/train')

        # returned transformed image
        tfm = transform['test'](img).unsqueeze_(0)
        return tfm.cuda() if cuda else tfm

    # create validation set
    if not os.path.isdir('output'):
        print('Splitting train folder into train/val...')
        _ = splitfolders.ratio(os.path.join(path, 'train'),
                               output='output', ratio=(.8, .2))

    # read data and preprocess
    train_holder = datasets.ImageFolder('output/train', transform=transform['train'])
    valid_holder = datasets.ImageFolder('output/val', transform=transform['test'])
    test_holder = datasets.ImageFolder(f'{path}/test', transform=transform['test'])

    # create loaders
    train_loader = DataLoader(train_holder, batch_size=n_batch, shuffle=True)
    valid_loader = DataLoader(valid_holder, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_holder, batch_size=n_batch, shuffle=True)

    return train_loader, valid_loader, test_loader, train_holder


def get_random_image(path):
    folder = path + '/' + random.choice(os.listdir(path))
    return Image.open(os.path.join(folder, random.choice(os.listdir(folder))))


def predict_landmarks(k: int, model, dictionary, img=None,
                      cuda=True, img_size=226, crop_size=224):
    """
    This function read the image file, applies appropriate transformations, predicts top K locations of images.

    Parameters:
         k (int):       Top locations to show
         cuda (bool):   Whether GPU is enabled or not (default=True)

    Returns:
        transformed image and pretty formatted location predictions
        :param img:
        :param dictionary:
        :param cuda:
        :param k:
        :param model:
        :param crop_size:
        :param img_size:
    """

    temp = get_random_image('data/train') if img is None else img
    transform = get_transforms(img_size, crop_size)

    img = temp.copy()
    img = transform['test'](img).unsqueeze_(0)
    img = img.cuda() if cuda else img

    # pass the model in evaluation mode
    model.eval()

    # get predictions
    output = model(img)

    # apply softmax to get probabilities
    output = F.softmax(output, dim=1)

    # show top K location predictions
    topk = torch.topk(output, k)
    prob = topk[0][0].detach().cpu().numpy()

    return temp, dictionary.get_content(topk[1].detach().cpu().numpy()[0]), prob
