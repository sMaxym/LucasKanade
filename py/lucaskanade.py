import scipy
import scipy.ndimage
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys

from sklearn.feature_extraction import image
from math import exp, ceil

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


# TODO: weight function
def bias(left_top, right_bottom, deriv_x, deriv_y, deriv_t, weight_func):
    bias_v = [0, 0]
    for row in range(left_top[0], right_bottom[0] + 1):
        for col in range(left_top[1], right_bottom[1] + 1):
            k = deriv_t[row, col] # * weight_func(row, col, offset_row, offset_col)
            bias_v[0] += k * deriv_x[row, col]
            bias_v[1] += k * deriv_y[row, col]
    return [-x for x in bias_v]


# TODO: weight function
def coeff_matrix(left_top, right_bottom, deriv_x, deriv_y, weight_func):
    mat = np.zeros((2, 2))
    for row in range(left_top[0], right_bottom[0] + 1):
        for col in range(left_top[1], right_bottom[1] + 1):
            xy_deriv_product = deriv_x[row, col] * deriv_y[row, col]
            mat[0, 0] += deriv_x[row, col] ** 2
            mat[0, 1] += xy_deriv_product
            mat[1, 0] += xy_deriv_product
            mat[1, 1] += deriv_y[row, col] ** 2
    return mat
    
def merge_imgs(imgs):
    imgs_num = len(imgs)
    w,h = imgs[0].shape[1], imgs[0].shape[0]
    img = np.zeros((h, w * imgs_num))
    for index, cur_img in enumerate(imgs):
            img[0:h, w * index:w * (index + 1)] = cur_img
    return img


if __name__ == '__main__':
    BLOCK_SHAPE_ROW = 5
    BLOCK_SHAPE_COL = 5
    
    sigma, ev_min, bias_prec = 1, 1, 4
    
    img0_orig = imageio.imread('./../data/monkey/monkey5_106.jpg')
    img1_orig = imageio.imread('./../data/monkey/monkey5_107.jpg') 
    # img0_orig = imageio.imread('./../data/grad00.jpg')
    # img1_orig = imageio.imread('./../data/grad01.jpg') 
    img_cur, img_next = rgb2gs(img0_orig), rgb2gs(img1_orig)
    num_rows, num_cols = img_cur.shape
    tracking_blocks = [(380, 100)]
    trans_vectors = [None] * len(tracking_blocks)
    
    img_cur = scipy.ndimage.gaussian_filter(img_cur, sigma)
    img_next = scipy.ndimage.gaussian_filter(img_next, sigma)
    deriv_x, deriv_y = np.zeros(img_cur.shape), np.zeros(img_cur.shape)
    # scipy.ndimage.sobel(img_cur, axis=1, output=deriv_x, mode="wrap")
    # scipy.ndimage.sobel(img_cur, axis=0, output=deriv_y, mode="wrap")
    # deriv_t = img_cur - img_next
    
    # TODO: add None handling 
    while not all((vec is not None and np.linalg.norm(vec) < bias_prec) for vec in trans_vectors):
        for index, block in enumerate(tracking_blocks):
            vec = trans_vectors.pop(0)
            if vec is None: continue
            row, col = block
            left_top = (row - BLOCK_SHAPE_ROW if row >= BLOCK_SHAPE_ROW else 0,
                        col - BLOCK_SHAPE_COL if col >= BLOCK_SHAPE_COL else 0)
            right_bottom = (row + BLOCK_SHAPE_ROW if row + BLOCK_SHAPE_ROW < num_rows else num_rows - 1,
                            col + BLOCK_SHAPE_COL if col + BLOCK_SHAPE_COL < num_cols else num_cols - 1)
            r_v_0 = int(round(vec[0]))
            r_v_1 = int(round(vec[1]))
            # img_cur[left_top[0] + r_v_1:right_bottom[0] + 1 + r_v_1, left_top[1] + r_v_0:right_bottom[1] + 1 + r_v_0] = img_cur[left_top[0]:right_bottom[0]+1, left_top[1]:right_bottom[1]+1]
            img_cur[left_top[1] + r_v_0:right_bottom[1] + 1 + r_v_0, left_top[0] + r_v_1:right_bottom[0] + 1 + r_v_1] = \
                img_cur[left_top[1]:right_bottom[1] + 1, left_top[0]:right_bottom[0] + 1]
            tracking_blocks[index] += np.array([r_v_0, r_v_1])
        print(tracking_blocks)
        scipy.ndimage.sobel(img_cur, axis=1, output=deriv_x, mode="wrap")
        scipy.ndimage.sobel(img_cur, axis=0, output=deriv_y, mode="wrap")
        deriv_t = img_cur - img_next
        plt.imshow(img_cur)
        plt.show()
            
        for block in tracking_blocks:
            row, col = block
            left_top = (row - BLOCK_SHAPE_ROW if row >= BLOCK_SHAPE_ROW else 0,
                        col - BLOCK_SHAPE_COL if col >= BLOCK_SHAPE_COL else 0)
            right_bottom = (row + BLOCK_SHAPE_ROW if row + BLOCK_SHAPE_ROW < num_rows else num_rows - 1,
                            col + BLOCK_SHAPE_COL if col + BLOCK_SHAPE_COL < num_cols else num_cols - 1)
            b = bias(left_top, right_bottom, deriv_x, deriv_y, deriv_t, gaussian2d)
            M = coeff_matrix(left_top, right_bottom, deriv_x, deriv_y, gaussian2d)
            min_ev = min(np.linalg.eigvals(M))
            if min_ev < ev_min:
                trans_vectors.append(None)
            else:
                transition = np.linalg.inv(M).dot(b)
                trans_vectors.append(transition)
    
    plt.imshow(img0_orig)
    for i, vec in enumerate(trans_vectors):
        if vec is None: continue
        plt.arrow(tracking_blocks[i][0], tracking_blocks[i][1], vec[0], vec[1],
                  fc="k", ec="k", head_width=10, head_length=10)
    plt.show()
    