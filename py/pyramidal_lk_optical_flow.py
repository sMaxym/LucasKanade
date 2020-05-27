import random

import numpy as np
from scipy.signal import convolve2d

import cv2 as cv

from math import exp, ceil


def gaussian2d(x, y, x_0=0, y_0=0, sigma_x=1, sigma_y=1, amp=1):
    return amp * exp(-((x - x_0) ** 2 / (2 * sigma_x ** 2) + (y - y_0) ** 2 / (2 * sigma_y ** 2)))


def identity_func(x, y, x_0=0, y_0=0):
    return 1


def rgb2gs(img):
    return np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])


def img_offset(val, step, max_val):
    offset = max_val - val
    return offset if offset < step else 0


def img_block(img, partirion=(1, 1)):
    block_w, block_h = img.shape[1] // partirion[1], img.shape[0] // partirion[0]
    prev_y, prev_x = 0, 0
    cur_y, cur_x = 0, 0
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
def bias(left_top, right_bottom, deriv_x, deriv_y, deriv_t, weight_func=gaussian2d):

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
    # print(right_bottom)
    # mat[0, 0] = np.sum((deriv_x[left_top[0]:right_bottom[0] + 1,
    #                     left_top[1]:right_bottom[1] + 1]) ** 2)
    # mat[0, 1] = mat[1, 0] = np.sum((deriv_y[left_top[0]:right_bottom[0] + 1,
    #                                 left_top[1]:right_bottom[1] + 1]) *
    #                                (deriv_x[left_top[0]:right_bottom[0] + 1,
    #                                 left_top[1]:right_bottom[1] + 1])
    #                                )
    #
    # mat[1, 1] = np.sum((deriv_y[left_top[0]:right_bottom[0] + 1,
    #                     left_top[1]:right_bottom[1] + 1]) ** 2)
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
    w, h = imgs[0].shape[1], imgs[0].shape[0]
    img = np.zeros((h, w * imgs_num))
    for index, cur_img in enumerate(imgs):
        img[0:h, w * index:w * (index + 1)] = cur_img
    return img


def LucasKanade(img_cur, img_next, features):
    Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
    Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image

    BLOCK_SHAPE_ROW = 1
    BLOCK_SHAPE_COL = 1

    sigma, ev_min, bias_prec = 1, 0.05, 2

    num_rows, num_cols = img_cur.shape

    Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
    Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction

    MAX_LEVEL = 1
    image0_pyramid = []
    image1_pyramid = []
    img1 = img_cur
    img2 = img_next

    for i in range(MAX_LEVEL):
        image0_pyramid.append(img1)
        image1_pyramid.append(img2)
        img1 = cv.pyrDown(img1)
        img2 = cv.pyrDown(img2)

    image0_pyramid = image0_pyramid[::-1]
    image1_pyramid = image1_pyramid[::-1]
    u = np.zeros(shape=(image0_pyramid[0].shape[0] // 2, image0_pyramid[0].shape[1] // 2))

    v = np.zeros(shape=(image1_pyramid[0].shape[0] // 2, image1_pyramid[0].shape[1] // 2))

    initial_tracking_blocks = np.copy(features)
    tracking_blocks = features // (2 ** MAX_LEVEL)

    for img_cur, img_next in zip(image0_pyramid, image1_pyramid):

        u = np.round(cv.pyrUp(u))
        v = np.round(cv.pyrUp(v))

        deriv_x = (convolve2d(img_cur, Gx) + convolve2d(img_next, Gx)) / 2

        deriv_y = (convolve2d(img_cur, Gy) + convolve2d(img_next, Gy)) / 2

        deriv_t = convolve2d(img_cur, Gt1) + convolve2d(img_next, Gt2)

        for index, block in enumerate(tracking_blocks):

            col, row = block

            left_top = (row - BLOCK_SHAPE_ROW if row >= BLOCK_SHAPE_ROW else 0,
                        col - BLOCK_SHAPE_COL if col >= BLOCK_SHAPE_COL else 0)
            right_bottom = (row + BLOCK_SHAPE_ROW + 1 if row + BLOCK_SHAPE_ROW + 1 < num_rows else num_rows - 1,
                            col + BLOCK_SHAPE_COL + 1 if col + BLOCK_SHAPE_COL + 1 < num_cols else num_cols - 1)
            try:
                b = bias(left_top, right_bottom, deriv_x, deriv_y, deriv_t, gaussian2d)
                M = coeff_matrix(left_top, right_bottom, deriv_x, deriv_y, gaussian2d)


            except:
                continue

            min_ev = min(np.linalg.eigvals(M))
            if min_ev >= ev_min:
                result = np.int32(np.matmul(np.linalg.pinv(M), b) / 2)
                # u[row, col] += result[0]
                # v[row, col] += result[1]
                # tracking_blocks[index] = np.array([col + u[row, col], row + v[row, col]])
                # print(result)
                tracking_blocks[index] += result

        tracking_blocks = np.round(tracking_blocks) * 2

    return initial_tracking_blocks, tracking_blocks
