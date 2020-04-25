import scipy
import numpy as np
import imageio
import matplotlib.pyplot as plt

import sys


def rgb2gs(img):
    return np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])



img = imageio.imread(sys.argv[1])

# img = np.mean(img, axis=2)
img = rgb2gs(img)
img = img.T

imageio.imwrite('test.jpg', img)

# plt.imshow(img)
# plt.show()