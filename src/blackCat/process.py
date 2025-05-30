from importlib.resources import files
import os
from skimage import io
from matplotlib import pyplot
from .methods import *
import random


def test_method(method, n=1, set=0, local=False, target=None):
    """
    Applies specified method to a set of test images

    Args:
        method: a method that takes an image as an argument
        n(int): number of photos to test, maximum of 3
        set(int): determines which images will be used, default will use random selection
        local(bool): used when running the program locally
        target(str): will save image to file if target is included
    
    Returns:
        set of images with method applied
    """

    if(set == 0):
        img_nums = random.sample(range(1,39), n)
    else:
        img_nums = list(range(set, set+n))

    if local:
        from tqdm import tqdm
        path = os.path.join('src', 'blackCat', 'images')
        paths = [os.path.join(path, f'img{i}.jpg') for i in img_nums]

    else:
        from tqdm.notebook import tqdm
        path = files('blackCat.images')
        paths = [
            path.joinpath(f'img{i}.jpg') for i in img_nums
        ]

    images = [io.imread(path) for path in paths]
    processed = [method(img) for img in images]


    with tqdm(total=1, desc="Processing image 1") as pbar1:
        processed.append(method(images[0]))
        pbar1.update(1)

    if(n > 1):
        with tqdm(total=1, desc="Processing image 2") as pbar2:
            processed.append(method(images[1]))
            pbar2.update(1)

    if(n > 2):
        with tqdm(total=1, desc="Processing image 3") as pbar3:
            processed.append(method(images[2]))
            pbar3.update(1)

    fig, ax = pyplot.subplots(n, 2, figsize=(3, 2 * n))

    if(n > 1):
        for i in range(n):
            ax[i, 0].imshow(images[i])
            ax[i, 0].axis('off')
            ax[i, 1].imshow(processed[i])
            ax[i, 1].axis('off')
    else:
        ax[0].imshow(images[0])
        ax[0].axis('off')
        ax[1].imshow(processed[0])
        ax[1].axis('off')
        pyplot.tight_layout(pad=0.5)
    
    if(target):
        fig.savefig(target, bbox_inches='tight', dpi=300)
    pyplot.show()

    return processed

def load_display(method, path, target):
    img = io.imread(path)
    processed = method(img)
    cmap = 'gray' if processed.ndim == 2 else None
    fig, ax = pyplot.subplots(1, 2, figsize=(3, 2))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].imshow(processed, cmap=cmap)
    ax[1].axis('off')
    pyplot.tight_layout(pad=0.5)

    if(target):
        fig.savefig(target, bbox_inches='tight', dpi=300)
    pyplot.show()

    return processed

# uncomment for local testing
# test_method(lab_clahe, 1, 0, True, 'img100.jpg')
# load_display(bpdhe, 'src/blackCat/images/img3.jpg')
# yolo('src/blackCat/images/img3.jpg')