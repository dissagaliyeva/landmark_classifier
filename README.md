# Landmark Classifier <a class="anchor" id="landmark"/>
This project contains the subset of <a href='https://www.kaggle.com/google/google-landmarks-dataset'>Google 
Landmarks</a> dataset on Kaggle. The objective of this project was to create both CLI and Jupyter Notebook examples
running CNN using various models. 

### Project Description
* 5000 images (50 folders, 100 images in each)
* Datasets can be found here (BE CAREFUL, automatic download): [landmark dataset](https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip)

Photo sharing and photo storage services like to have location data for each photo that is uploaded. 
With the location data, these services can build advanced features, such as automatic suggestion of 
relevant tags or automatic photo organization, which help provide a compelling user experience. 
Although a photo's location can often be obtained by looking at the photo's metadata, many photos 
uploaded to these services will not have location metadata available. This can happen when, for 
example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed 
due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect 
and classify a discernible landmark in the image. Given the large number of landmarks across 
the world and the immense volume of images that are uploaded to photo sharing services, using 
human judgement to classify these landmarks would not be feasible.


### Installation

Clone the GitHub repository and use the following commands:
```
git clone https://github.com/dissagaliyeva/landmark_classifier
conda create --name landmark_classifier 
pip install -r requirements.txt
conda install -c fastai fastai
```

### Usage 

- **Run a single-image prediction with default model (resnet34)** 

There are 12 different images from 4 classes to choose from in [images](https://github.com/dissagaliyeva/landmark_classifier/tree/master/images) folder. 

```
python run.py
```

- **Train your own model**

If you want to train your own model, make sure to specify:
1) Custom (default=False):                    -c True
2) Data path (default='data')                 -p "data"
3) Model (default=resnet34, vgg16, resnet18): -m "resnet34"
4) Train (default=pytorch, fastai):           -t "pytorch" 
5) Batch size (default=16):                   -b 16
6) Epochs (default=20):                       -e 20
7) Optimizer (default=adam, sgd, adagrad)     -o "adam"

Example: 

```
python -c True

python -c True -m "resnet18" -t "fastai" -b 32 -e 50 -o "adagrad"
```

