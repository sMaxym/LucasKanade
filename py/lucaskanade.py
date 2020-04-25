import scipy
import numpy as np
import imageio
import matplotlib.pyplot as plt

import sys

img = imageio.imread(sys.argv[1])

img = np.mean(img, axis=2)
img = img.T

imageio.imwrite('test.jpg', img)

# plt.imshow(img)
# plt.show()