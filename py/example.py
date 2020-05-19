from scipy.signal import convolve2d
from skimage.transform import pyramid_gaussian

import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2 as cv
from math import exp, ceil


def lucas_kanade(img1, img2, pyramid_level, block_size=(7, 7)):
    first_pyramid = pyramid_gaussian(img1, max_layer=pyramid_level,
                                     multichannel=False)
    second_pyramid = pyramid_gaussian(img2, max_layer=pyramid_level,
                                     multichannel=False)

    pyramids_iter = zip(first_pyramid[::-1], second_pyramid[::-1])
    v = np.zeros(shape=first_pyramid[pyramid_level - 1].shape)
    u = np.zeros(shape=first_pyramid[pyramid_level - 1].shape)
    for first_pyramid_image, second_pyramid_image in pyramids_iter:
        pass





