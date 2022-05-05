# USAGE
# python run.py

import re
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from utils.visualize.visualize import show_image


# give info about the default values
print("""
Hi! Thanks for running the Landmark Classifier app. 

By default, the app tests a ResNet34 model on Vienna City Hall image. 

================= Select Own Image ================

If you want to select a different image (there are 4 folders to choose from in "images"), specify it as following:
python run.py -i "path/to/image"

================= Train Your Model ================

If you want to train your own model, make sure to specify:
1) Custom (default=False):                    -c True
2) Model (default=resnet34, vgg16, resnet18): -m "resnet34"
3) Train (default=pytorch, fastai):           -t "pytorch" 
4) Batch size (default=16):                   -b 16
5) Epochs (default=20):                       -e 20
6) Optimizer (default=adam, sgd, adagrad)     -o "adam"

Example: 
python -c True -m "resnet18" -t "pytorch" -b 16 -e 100 -o "adagrad"

""")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='images/19.Vienna_City_Hall/33fdae363340e364.jpg',
                type=str, help='Path to an image to test')

# train custom model
ap.add_argument('-c', '--custom', required=False, default=False,
                type=bool, help='Whether to train a custom model')
ap.add_argument('-m', '--model', required=False, default='resnet34',
                type=str, help='CNN model to run (vgg16, resnet18, resnet34)')
ap.add_argument('-t', '--train', required=False, default='pytorch',
                type=str, help='Training option between PyTorch & FastAI')
ap.add_argument('-b', '--batch', required=False, default=16,
                type=int, help='Batch size for training')
ap.add_argument('-e', '--epochs', required=False, default=20,
                type=int, help='Number of epochs to train the model')
ap.add_argument('-o', '--optim', required=False, default='Adam',
                type=str, help='Optimizer to use (adam, sgd, adagrad)')

args = vars(ap.parse_args())

if args['custom']:
    print('Training a model...')

else:
    print('Opening the image...')
    img   = Image.open(args['image'])
    title = args['image'].split('/')[1].replace('_', ' ')
    title = ' '.join(re.findall('[A-Za-z]+', title))

    show_image(img, title)

