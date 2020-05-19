import scipy
import scipy.ndimage
from scipy.signal import convolve2d
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys


def rgb2gs(img):
    return np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])


if __name__ == "__main__":
    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image
    img0_orig = rgb2gs(imageio.imread('./../data/frame_00_delay-1s.gif'))
    img1_orig = rgb2gs(imageio.imread('./../data/frame_01_delay-1s.gif'))
    It = convolve2d(img0_orig, Gt1) + convolve2d(img1_orig, Gt2)

    plt.imshow(It)
    plt.show()

    plt.imshow(img1_orig - img0_orig)
    plt.show()
