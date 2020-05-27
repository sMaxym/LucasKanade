import random

from py.pyramidal_lk_optical_flow import LucasKanade
import numpy as np
import cv2 as cv


if __name__ == '__main__':

    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)
    color = np.random.randint(0, 255, (100, 3))
    cap = cv.VideoCapture("../data/max_video.mp4")
    ret, old_frame = cap.read()

    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

    # Finding features
    tracking_blocks = cv.goodFeaturesToTrack(old_gray, mask=None,
                                             **feature_params)

    x = np.zeros(shape=(tracking_blocks.shape[0], tracking_blocks.shape[2]))
    for i in range(tracking_blocks.shape[0]):
        x[i] = tracking_blocks[i, 0]
    tracking_blocks = np.int32(x)

    x, y = (350, 350)
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    second_frame = np.zeros_like(old_frame)
    mask = np.zeros_like(old_frame)
    width, height, layer = old_frame.shape
    out_video = cv.VideoWriter('../outx.avi', cv.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
    while True:
        try:
            ret, frame = cap.read()

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            initial_tracking_blocks, tracking_blocks = LucasKanade(old_gray, frame_gray,
                                                                           tracking_blocks)
            vectors = initial_tracking_blocks - tracking_blocks
            super_vector = np.zeros(shape=(1, 2))

            for vector in vectors:
                super_vector += vector

            super_vector //= (vectors.shape[0])

            mask = cv.line(mask, (x, y), (x - int(super_vector[0][0]),
                                          y - int(super_vector[0][1])),
                           color[random.randint(1, 99)].tolist(), 2)

            frame = cv.circle(frame, (x, y), 5,
                              color[random.randint(1, 99)].tolist(), -1)
            x -= int(super_vector[0][0])
            y -= int(super_vector[0][1])

            img = cv.add(frame, mask)

            cv.imshow('frame', img)
            out_video.write(img)

            wait_key = cv.waitKey(100)
            if wait_key == 25:
                break

            old_gray = frame_gray.copy()
        except Exception as e:
            print(e)
            break
    out_video.release()
    cv.destroyAllWindows()
    cap.release()
