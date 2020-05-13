import scipy
import scipy.ndimage
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

from sklearn.feature_extraction import image
from math import exp, ceil


PART_X, PART_Y = 20, 20
PARTITION = (PART_Y, PART_X)
BLOCK_SHAPE = (20, 20)

def gaussian2d(x, y, x_0=0, y_0=0, sigma_x=1, sigma_y=1, amp=1):
    return amp * exp(-( (x - x_0) ** 2 / (2 * sigma_x ** 2) + (y - y_0) ** 2 / (2 * sigma_y ** 2) ))

def identity_func(x, y, x_0=0, y_0=0):
    return 1

def rgb2gs(img):
    return np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])

def img_offset(val, step, max_val):
    offset = max_val - val 
    return offset if offset < step else 0

def img_block(img, partirion=(1, 1)):
    block_w, block_h = img.shape[1] // partirion[1], img.shape[0] // partirion[0]
    prev_y, prev_x = 0,0
    cur_y, cur_x = 0,0
    while (cur_y, cur_x) != img.shape:
        cur_y, cur_x = prev_y + block_h, prev_x + block_w
        cur_x += img_offset(cur_x, block_w, img.shape[1])
        cur_y += img_offset(cur_y, block_h, img.shape[0])
        if cur_x > img.shape[1]:
            cur_x %= img.shape[1]
            cur_y += block_h
        yield img[prev_y:cur_y, prev_x:cur_x]
        prev_x = cur_x
        if prev_x >= img.shape[1]:
            prev_x %= img.shape[1]
            prev_y += block_h
            
def img_block_pixelvice(img, block_shape=(5, 5)):
    rows, cols = img.shape
    block_rows, block_cols = block_shape 
    col_parts = ceil(cols / block_cols)
    row_parts = ceil(rows / block_rows)
    for row in range(row_parts):
        for col in range(col_parts):
            diff_row, diff_col = block_rows, block_cols
            if (row + 1) * block_rows >= rows:
                diff_row = rows - row * block_rows
            if (col + 1) * block_cols >= cols:
                diff_col = cols - col * block_cols
            yield img[row * block_rows:row * block_rows + diff_row, col * block_cols:col * block_cols + diff_col]


def bias(deriv_x, deriv_y, deriv_t, weight_func):
    n_rows, n_cols = deriv_t.shape
    offset_row, offset_col = n_rows // 2, n_cols // 2
    bias_v = [0, 0]
    for row in range(n_rows):
        for col in range(n_cols):
            k = deriv_t[row, col] * weight_func(row, col, offset_row, offset_col)
            bias_v[0] += k * deriv_x[row, col]
            bias_v[1] += k * deriv_y[row, col]
    return [-x for x in bias_v]

def coeff_matrix(deriv_x, deriv_y, weight_func):
    n_rows, n_cols = deriv_x.shape
    offset_row, offset_col = n_rows // 2, n_cols // 2
    mat = np.zeros((2, 2))
    for row in range(n_rows):
        for col in range(n_cols):
            w = weight_func(row, col, offset_row, offset_col)
            xy_deriv_product = deriv_x[row, col] * deriv_y[row, col]
            mat[0, 0] += w * deriv_x[row, col] ** 2
            mat[0, 1] += w * xy_deriv_product
            mat[1, 0] += w * xy_deriv_product
            mat[1, 1] += w * deriv_y[row, col] ** 2
    return mat

def show_merged(imgs):
    """
    Sizes of images should be the same
    """
    imgs_num = len(imgs)
    w,h = imgs[0].shape[1], imgs[0].shape[0]
    img = np.zeros((h, w * imgs_num))
    for index, cur_img in enumerate(imgs):
            img[0:h, w * index:w * (index + 1)] = cur_img
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    img0_orig = imageio.imread('./../data/monkey/monkey5_103.jpg')
    img1_orig = imageio.imread('./../data/monkey/monkey5_104.jpg') 
    img0, img1 = rgb2gs(img0_orig), rgb2gs(img1_orig)
    
    # img0 = np.array([[0, 0, 0],
    #                  [4, 1, 0],
    #                  [5, 3, 0]])

    # img1 = np.array([[0, 4, 2],
    #                  [0, 4, 2],
    #                  [0, 1, 0]])
    
    res = []
    
    sigma = 1.5
    
    img0_ = scipy.ndimage.gaussian_filter(img0, sigma)
    img1_ = scipy.ndimage.gaussian_filter(img1, sigma)
    
    time_spart = np.matrix([img0_.flatten(), img1_.flatten()])
    time_spart = scipy.ndimage.gaussian_filter1d(time_spart, sigma / 3, axis=0)
    
    img0_ = time_spart[0,:].reshape(img0.shape)
    img1_ = time_spart[1,:].reshape(img0.shape)
    show_merged([img0, img1, img0_, img1_])
    
    
    block_pairs = zip(img_block_pixelvice(img0_, BLOCK_SHAPE), img_block_pixelvice(img1_, BLOCK_SHAPE))
    for img, next_img in block_pairs:
        deriv_x, deriv_y = np.zeros(img.shape), np.zeros(img.shape)
        deriv_t = img - next_img
        scipy.ndimage.sobel(img, axis=1, output=deriv_x, mode="constant")
        scipy.ndimage.sobel(img, axis=0, output=deriv_y, mode="constant")
        b = bias(deriv_x, deriv_y, deriv_t, gaussian2d)
        M = coeff_matrix(deriv_x, deriv_y, gaussian2d)
        trans = np.linalg.pinv(M).dot(b)
        
        res.append(trans)
        
    for item in res:
        print(item)
        
    plt.imshow(img0_orig)
    rows, cols = img0_orig.shape[0], img0_orig.shape[1]
    block_rows, block_cols = BLOCK_SHAPE
    col_parts = ceil(cols / block_cols)
    row_parts = ceil(rows / block_rows)
    for row in range(row_parts):
        for col in range(col_parts):
            diff_row, diff_col = block_rows, block_cols
            if (row + 1) * block_rows >= rows:
                diff_row = rows - row * block_rows
            if (col + 1) * block_cols >= cols:
                diff_col = cols - col * block_cols
            diff_row //= 2
            diff_col //= 2
            plt.arrow(col * block_cols + diff_col, row * block_rows + diff_row, res[row * col_parts + col][1], res[row * col_parts + col][0], fc="k", ec="k",head_width=2, head_length=2 )
    plt.show()
    