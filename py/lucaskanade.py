import scipy
import imageio
import matplotlib.pyplot as plt

import sys

img = imageio.imread(sys.argv[1])

print(img.shape)

# plt.imshow(img)
# plt.show()