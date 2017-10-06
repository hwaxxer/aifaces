from scipy.misc import imread, imresize
from skimage.color import rgba2rgb
import numpy as np

def imcrop_square(img):
    size = np.min(img.shape[:2])
    extra = img.shape[:2] - size
    crop = img
    for i in np.flatnonzero(extra):
        crop = np.take(crop, extra[i] // 2 + np.r_[:size], axis=i)
    return crop

def preprocess_file(f, shape):
    img = imread(f)
    if len(img.shape) == 3 and img.shape[-1] > 3:
        # Remove alpha
        img = rgba2rgb(img)

    # Just stack the image if it only has 1 channel
    img = img if len(img.shape) == 3 else np.stack((img,)*3, axis=-1)
    img = imcrop_square(img)
    img = imresize(img, shape)
    return img

def split_image(img):
    xs = []
    ys = []

    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            xs.append([row_i, col_i])
            ys.append(img[row_i, col_i])

    return np.array(xs), np.array(ys)
