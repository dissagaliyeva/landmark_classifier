import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import re
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import splitfolders

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def create_loaders(n_batch, path='data', batch=True,
                   cuda=False, img_size=226, crop_size=224) -> tuple:
    """

    :param crop_size:
    :param img_size:
    :param n_batch:
    :param path:
    :param batch:
    :param cuda:
    :return:
    """

    # define transformations
    transform = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(crop_size),
            transforms.RandomRotation((-10, 10)),
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
    test_holder  = datasets.ImageFolder('data/test', transform=transform['test'])

    # create loaders
    train_loader = DataLoader(train_holder, batch_size=n_batch, shuffle=True)
    valid_loader = DataLoader(valid_holder, batch_size=n_batch, shuffle=True)
    test_loader  = DataLoader(test_holder, batch_size=n_batch, shuffle=True)

    return train_loader, valid_loader, test_loader, train_holder


def get_random_image(path):
    folder = path + '/' + random.choice(os.listdir(path))
    return Image.open(os.path.join(folder, random.choice(os.listdir(folder))))


# create dictionary that stores vocabulary
class Dictionary:
    """
    A class to represent a dictionary.

        Attributes:
            dataset: ImageFolder that stores images from training folder

        Methods:
            create_inverse():   Creates the inverse of the dictionary. Stores "int: class" pairs
            simple_print(idx):  Shows first N integers and their respective classes
            get_item(item):     Returns the class name at specific index
            get_content(index): Returns string representation of a list of indices
    """

    def __init__(self, dataset):
        """
        Instantiates the inverse dictionary.

        Parameters:
            dataset: ImageFolder that stores images from training folder
        """
        self.dataset = dataset
        self.inverse_dict = self.create_inverse()

    def create_inverse(self) -> dict:
        """
        Creates the inverse of the dictionary. Stores "int: class" pairs

        Returns: An inverse dictionary (int: class)
        """
        return dict((v, k) for k, v in self.dataset.class_to_idx.items())

    def simple_print(self, idx=50):
        """
        Shows first N integers and their respective classes.

        Parameters:
            idx (int): Number of indices to show

        Returns: None
        """
        # show all classes from training folder
        print('\t\t\t\t\t\tClasses & Indexes')
        for i, v in enumerate(self.dataset.class_to_idx.values()):
            if i == idx: break
            print(f'{v}:\t{self.get_content(v)}')

    def get_item(self, item: int) -> str:
        """
        Returns a class name from a dictionary.

        Parameters:
            item (int): Index value to lookup

        Returns: Class name
        """
        return self.inverse_dict[item]

    def get_content(self, index) -> str:
        """
        Gets the indices and outputs a beautiful representation of class names.

        Parameters:
            index (list or int): List or index values to lookup

        Returns: string representation separated by commas
        """

        # remove leading digits and underscores
        make_prettier = lambda x: ' '.join(re.findall('[A-Za-z]+', self.get_item(x)))

        # check if it's a single index
        if type(index) == int:
            return make_prettier(index)

        # return comma-separated representation
        return ', '.join([make_prettier(x) for x in index])


def visualize(dictionary, loader=None, single=True):
    """
    Create single and batch visualizations.

    Parameters:
        loader:             Instance of DataLoader to iterate through
        dictionary (class): Previously created Dictionary class
        single (bool):      Show a single image or not. Default is False
    """

    # create converters for images and labels
    # convert = lambda x: np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)
    convert_label = lambda x: str(x.item())

    # transform single images and their labels
    def show_single(image, lbl, index=0):
        # unnormalize = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
        denorm = transforms.Normalize(
            mean=[-m / s for m, s in zip(MEAN, STD)],
            std=[1.0 / s for s in STD]
        )
        image = (denorm(image=image)["image"] * 255).astype(np.uint8)

        # image = unnormalize(image[index, :])
        lbl = convert_label(lbl[index])   # get the label from dictionary
        return image, dictionary.get_content(int(lbl))

    # iterate through one or batch of images
    images, labels = next(iter(loader))
    img_len = len(images)

    # show single image
    if single:
        i, l = show_single(images, labels)
        plt.title(l, fontsize=20)
        plt.grid(False)
        plt.imshow(i)

    else:
        # create a figure to show img_len batch of images
        fig = plt.figure(figsize=(30, 10))
        fig.tight_layout(pad=3.0)
        fig.suptitle(f'Sample batch of {img_len}', fontsize=40, y=0.55)

        for idx in range(img_len):
            ax = fig.add_subplot(2, int(img_len / 2), idx + 1, xticks=[], yticks=[])
            image, label = show_single(images, labels, idx)
            ax.set_title(label, fontsize=15)
            plt.grid(False)
            ax.imshow(image)
    plt.show()

