from scipy.signal import convolve2d
from skimage.transform import pyramid_gaussian

import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2 as cv
from math import exp, ceil


class LucasKanade:
    def __init__(self):
        pass

    def gaussian2d(self, x, y, x_0=0, y_0=0, sigma_x=1, sigma_y=1, amp=1):
        return amp * exp(-( (x - x_0) ** 2 / (2 * sigma_x ** 2) + (y - y_0) ** 2 / (2 * sigma_y ** 2) ))

    def identity_func(self, x, y, x_0=0, y_0=0):
        return 1

    def rgb2gs(self, img):
        return np.dot(img[..., :3], [0.2125, 0.7154, 0.0721])

    def img_offset(self, val, step, max_val):
        offset = max_val - val
        return offset if offset < step else 0

    def img_block(self, img, partirion=(1, 1)):
        block_w, block_h = img.shape[1] // partirion[1], img.shape[0] // partirion[0]
        prev_y, prev_x = 0,0
        cur_y, cur_x = 0,0
        while (cur_y, cur_x) != img.shape:
            cur_y, cur_x = prev_y + block_h, prev_x + block_w
            cur_x += self.img_offset(cur_x, block_w, img.shape[1])
            cur_y += self.img_offset(cur_y, block_h, img.shape[0])
            if cur_x > img.shape[1]:
                cur_x %= img.shape[1]
                cur_y += block_h
            yield img[prev_y:cur_y, prev_x:cur_x]
            prev_x = cur_x
            if prev_x >= img.shape[1]:
                prev_x %= img.shape[1]
                prev_y += block_h


    def img_block_pixelvice(self, img, block_shape=(5, 5)):
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


    def bias(self, deriv_x, deriv_y, deriv_t, weight_func):
        n_rows, n_cols = deriv_x.shape
        offset_row, offset_col = n_rows // 2, n_cols // 2
        bias_v = [0, 0]
        for row in range(n_rows):
            for col in range(n_cols):
                k = deriv_t[row, col] * weight_func(row, col, offset_row, offset_col)
                bias_v[0] += k * deriv_x[row, col]
                bias_v[1] += k * deriv_y[row, col]
        return [-x for x in bias_v]


    def coeff_matrix(self, deriv_x, deriv_y, weight_func):
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

    def show_merged(self, imgs):
        """
        Sizes of images should be the same
        """
        imgs_num = len(imgs)
        w, h = imgs[0].shape[1], imgs[0].shape[0]
        img = np.zeros((h, w * imgs_num))
        for index, cur_img in enumerate(imgs):
            img[0:h, w * index:w * (index + 1)] = cur_img
        plt.imshow(img)
        plt.show()

    def pyramidal_implement(self, img0, img1, PYRAMID_LEVEL=2):

        img0 = self.rgb2gs(img0)
        img1 = self.rgb2gs(img1)
        first_img_pyramid = list(pyramid_gaussian(img0, max_layer=PYRAMID_LEVEL,
                                                   multichannel=False))
        second_img_pyramid = list(pyramid_gaussian(img1, max_layer=PYRAMID_LEVEL,
                                                    multichannel=False))
        # print(img0.shape)
        i = 7
        u = np.zeros(shape=first_img_pyramid[PYRAMID_LEVEL - 1].shape)
        v = np.zeros(shape=first_img_pyramid[PYRAMID_LEVEL - 1].shape)
        level_results = [(0., 0.)]*10000
        for level_img, level_next_img in zip(first_img_pyramid[::1], second_img_pyramid[::1]):
            # print(level_img.shape)
            level_result = self.implement(level_img, level_next_img, (i, i))
            print(len(level_result))
            # for k in range(len(level_result)):
            #     level_results[k] += level_result[k]
            # print(len(level_result))
            self.show_results(level_img, level_result, (i, i))
            i *= 2
            # self.calculate(level_result, u, v, (i, i))
            break
            # u = cv.resize(u, dsize=(u.shape[0]*2, u.shape[1]*2))
            # v = cv.resize(v, dsize=(u.shape[0]*2, u.shape[1]*2))
            # plt.imshow(u)
            # plt.show()
            # plt.imshow(v)
            # plt.show()
            # plt.imshow(img0)
            # print(len(level_result))
            # self.show_results(level_img, level_result)
            # g_L = 2*(g_L + level_result)
            # # plt.imshow(level_img)
            # # plt.show()
        # return g_L

    @staticmethod
    def show_results(img, res, BLOCK_SHAPE=(7, 7)):
        plt.imshow(img)
        rows, cols = img.shape[0], img.shape[1]
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
                if res[row * col_parts + col][1] < 1 and res[row * col_parts + col][0] < 1:
                    continue
                if diff_row < 0.1 or diff_col < 0.1:
                    continue
                plt.arrow(col * block_cols + diff_col, row * block_rows + diff_row, res[row * col_parts + col][1],
                          res[row * col_parts + col][0], fc="k", ec="k", head_width=2, head_length=2)
        plt.show()

    def calculate(self, res, u, v, BLOCK_SHAPE=(7, 7)):
        rows, cols = u.shape[0], u.shape[1]
        print(u.shape)
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
                try:
                    if res[row * col_parts + col][1] < 1 and res[row * col_parts + col][0] < 1:
                        continue
                except:
                    continue
                # if diff_row < 0.1 or diff_col < 0.1:
                #     continue
                try:
                    u[row, col] += res[row * col_parts + col][0]
                    v[row, col] += res[row * col_parts + col][1]
                except:
                    pass

                # plt.arrow(col * block_cols + diff_col, row * block_rows + diff_row, res[row * col_parts + col][1],
                #           res[row * col_parts + col][0], fc="k", ec="k", head_width=2, head_length=2)

    def implement(self, img0_orig, img1_orig, BLOCK_SHAPE = (7, 7), sigma = 1.4):

        Gx = np.reshape(np.asarray([[-1, 1], [-1, 1]]), (2, 2))  # for image 1 and image 2 in x direction
        Gy = np.reshape(np.asarray([[-1, -1], [1, 1]]), (2, 2))  # for image 1 and image 2 in y direction
        Gt1 = np.reshape(np.asarray([[-1, -1], [-1, -1]]), (2, 2))  # for 1st image
        Gt2 = np.reshape(np.asarray([[1, 1], [1, 1]]), (2, 2))  # for 2nd image
        # img0_orig = imageio.imread('./../data/basketball1.png')
        # img1_orig = imageio.imread('./../data/basketball2.png')

        # img0, img1 = self.rgb2gs(img0_orig), self.rgb2gs(img1_orig)
        img0, img1 = img0_orig, img1_orig
        # plt.imshow(img0)

        # plt.show()
        # img0 = np.array([[0, 0, 0],
        #                  [4, 1, 0],
        #                  [5, 3, 0]])

        # img1 = np.array([[0, 4, 2],
        #                  [0, 4, 2],
        #                  [0, 1, 0]])

        res = []



        # img0_ = scipy.ndimage.gaussian_filter(img0, sigma)
        # img1_ = scipy.ndimage.gaussian_filter(img1, sigma)

        # time_spart = np.matrix([img0.flatten(), img1.flatten(),
        #                         img2.flatten(), img3.flatten()])
        # time_spart = scipy.ndimage.gaussian_filter1d(time_spart, sigma / 3, axis=0)

        # img0_ = time_spart[0,:].reshape(img0.shape)
        # img1_ = time_spart[1,:].reshape(img0.shape)
        # print(img0.shape)
        # show_merged([img0, img1, img0_, img1_])
        deriv_x = (convolve2d(img0, Gx) + convolve2d(img1, Gx)) / 2
        deriv_y = (convolve2d(img0, Gy) + convolve2d(img1, Gy)) / 2
        deriv_t = convolve2d(img0, Gt1) + convolve2d(img1, Gt2)
        # block_pairs = zip(img_block_pixelvice(img0, BLOCK_SHAPE), img_block_pixelvice(img1, BLOCK_SHAPE))
        block_pair = zip(self.img_block_pixelvice(deriv_x, BLOCK_SHAPE),
                         self.img_block_pixelvice(deriv_y, BLOCK_SHAPE),
                         self.img_block_pixelvice(deriv_t, BLOCK_SHAPE))
        # for img, next_img in block_pairs:
        for deriv_x, deriv_y, deriv_t in block_pair:
            # deriv_x, deriv_y = np.zeros(img.shape), np.zeros(img.shape)
            # deriv_t = next_img - img
            # print(deriv_t.shape)
            # deriv_x = convolve2d(img, Gx)
            # deriv_y = convolve2d(img, Gy)
            # deriv_t = convolve2d(next_img, Gt1) + convolve2d(img, Gt2)
            # print(deriv_x.shape, deriv_y.shape, deriv_t.shape)
            # scipy.ndimage.sobel(img, axis=1, output=deriv_x, mode="constant")
            # scipy.ndimage.sobel(img, axis=0, output=deriv_y, mode="constant")
            b = self.bias(deriv_x, deriv_y, deriv_t, self.gaussian2d)
            M = self.coeff_matrix(deriv_x, deriv_y, self.gaussian2d)
            # trans = np.linalg.pinv(M.transpose().dot(M)).dot(M.transpose().dot(b))
            trans = np.linalg.pinv(M).dot(b)
            eigvals = np.linalg.eigvals(M)

            if abs(trans[0] - min(eigvals)) < 0.5 and abs(trans[1] - min(eigvals)) < 0.5:
                res.append(np.array([0, 0]))
            else:
                res.append(trans)
        #
        # for item in res:
        #     item *= 1
        #     print(item)


        return res


if __name__ == '__main__':
    lk = LucasKanade()
    cap = cv.VideoCapture("../data/video.mp4")

    ret, frame = cap.read()

    # old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #
    # plt.imshow(old_gray)
    # plt.show()
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    while True:
        ret, new_frame = cap.read()
        # new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2GRAY)
        x = lk.implement(lk.rgb2gs(frame), lk.rgb2gs(new_frame))
        # x = lk.pyramidal_implement(frame, new_frame)
        lk.show_results(frame, x)
        break
        frame = new_frame

        # plt.imshow(img)
        # plt.show()


        #
        # # exit(0)
        # # img0 = np.array([[0, 0, 0],
        # #                  [4, 1, 0],
        # #                  [5, 3, 0]])**2
        # #
        # # img1 = np.array([[0, 4, 2],
        # #                  [0, 4, 2],
        # #                  [0, 1, 0]])**2
        #
        # # res = [np.array([0., 0.])]*100
        # # print(len(res))
        # PYRAMID_LEVEL = 0
        # tau = 0.01
        # print("XX")

        # # first_img_pyramid[0] = img0_orig
        # # second_img_pyramid[0] = img1_orig
        # # plt.imshow(first_img_pyramid[1])
        # # plt.show()
        # # print(rgb2gs(img0_orig).shape)
        # # for p in first_img_pyramid:
        # #     print(p.shape)

        #
        #

        #     # print("Hello")
        #     # print("LVL", level_img.shape)
        #     res = np.zeros(shape=(100, 2))
        #     # print(level_img.shape)
        #     PARTITION = [int(PARTITION[0]*2), int(PARTITION[1]*2)]
        #     print(PARTITION)
        #
        #     index = -1
        #
