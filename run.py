# USAGE
# python run.py
import os.path
import re
import argparse
from PIL import Image
from utils.models import models

import warnings
warnings.filterwarnings('ignore')


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
2) Data path (default='data')                 -p "data"
3) Model (default=resnet34, vgg16, resnet18): -m "resnet34"
4) Train (default=pytorch, fastai):           -t "pytorch" 
5) Batch size (default=16):                   -b 16
6) Epochs (default=20):                       -e 20
7) Optimizer (default=adam, sgd, adagrad)     -o "adam"

Example: 
python -c True 

python -c True -m "resnet18" -t "fastai" -b 32 -e 50 -o "adagrad"

""")

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=False, default='images/19.Vienna_City_Hall/33fdae363340e364.jpg',
                type=str, help='Path to an image to test')

# train custom model
ap.add_argument('-c', '--custom', required=False, default=False,
                type=bool, help='Whether to train a custom model (default=False)')

ap.add_argument('-p', '--path', required=False, default='data/',
                type=str, help='Path to data folder. It should have train/test '
                               'or train/val/test folders (default="data/")')

ap.add_argument('-m', '--model', required=False, default='resnet34',
                type=str, help='CNN model to run (options: vgg16, resnet18, resnet34) '
                               '(default="resnet34)"')

ap.add_argument('-t', '--train', required=False, default='pytorch',
                type=str, help='Training option between PyTorch & FastAI (default="pytorch")')

ap.add_argument('-b', '--batch', required=False, default=16,
                type=int, help='Batch size for training (default=16)')

ap.add_argument('-e', '--epochs', required=False, default=50,
                type=int, help='Number of epochs to train the model (default=20)')

ap.add_argument('-o', '--optim', required=False, default='Adam',
                type=str, help='Optimizer to use (options: adam, sgd, adagrad) (default="adam")')

args = vars(ap.parse_args())

if args['custom']:
    print('Training a model...')
    models.train(path=args['path'], model=args['model'],
                 mode=args['train'], batch_size=args['batch'],
                 epochs=args['epochs'], optim=args['optim'])

else:
    path  = args['image']
    model = args['model']

    # verify the image and model exist
    assert os.path.exists(path), f'{path} does not exist.'
    assert model.lower() in ['vgg16', 'resnet14', 'resnet34'], f'Please choose one of the following models: ' \
                                                               f'vgg16, resnet18, resnet34'

    img   = Image.open(path)
    title = path.split('/')[1].replace('_', ' ')
    title = ' '.join(re.findall('[A-Za-z]+', title))
    models.predict_image(img, title, model_name=model)



























