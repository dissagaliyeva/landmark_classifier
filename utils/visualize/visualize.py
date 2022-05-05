import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.cuda

from utils.preprocess.preprocess import predict_landmarks


def visualize(dictionary, loader=None, single=True):
    """
    Create single and batch visualizations.

    Parameters:
        loader:             Instance of DataLoader to iterate through
        dictionary (class): Previously created Dictionary class
        single (bool):      Show a single image or not. Default is False
    """

    # create converters for images and labels
    convert = lambda x: np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)
    convert_label = lambda x: str(x.item())

    # transform single images and their labels
    def show_single(image, lbl, index=0):
        image = convert(image[index, :])  # transform the image
        lbl = convert_label(lbl[index])  # get the label from dictionary
        return image, dictionary.get_content(int(lbl))

    # iterate through one or batch of images
    images, labels = next(iter(loader))
    img_len = 8

    # show single image
    if single:
        i, l = show_single(images, labels)
        plt.figure(figsize=(9, 5))
        plt.title(l, fontsize=20)
        plt.imshow(i)
    else:
        # create a figure to show img_len batch of images
        fig = plt.figure(figsize=(20, 8))

        for idx in range(img_len):
            ax = fig.add_subplot(1, 8, idx + 1, xticks=[], yticks=[])
            image, label = show_single(images, labels, idx)
            ax.imshow(image)
            ax.set_title(label, fontsize=10, wrap=True)
    plt.show()


def show_image(image, name=None, prediction=None):
    plt.figure(figsize=(9, 5))
    plt.grid(False)
    plt.imshow(image)

    if name:
        plt.title(name.title(), fontsize=16)

    plt.xticks([])
    plt.yticks([])
    plt.show()


def suggest_locations(img, model, dictionary, k=1, name=None):
    """
    This function shows the final image and associated K predictions.

    """
    # get cuda condition
    cuda = torch.cuda.is_available()

    # get landmark predictions
    img, predicted_landmarks, probs = predict_landmarks(img=img, k=k, dictionary=dictionary,
                                                        model=model, cuda=cuda)

    preds = pd.DataFrame({'loc': predicted_landmarks.split(','),
                          'prob': probs})
    preds['prob'] = preds['prob'].apply(lambda x: float(format(x, '.3f')))
    preds = preds.sort_values('prob', ascending=False)

    # normalize the image and show the predictions
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img.resize((226, 226)))
    ax[0].grid(False)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # show probabilities
    ax[1].hist(preds['loc'], weights=preds['prob'])

    # add title and show
    title = 'Single-image prediction '
    if name:
        title += f'for {name}'

    fig.suptitle(title, fontsize=17)
    plt.tight_layout()
    plt.show()

