import random

import numpy as np
import cv2

cap = cv2.VideoCapture('../data/max_video.mp4')

# params for corner detection
feature_params = dict(maxCorners=10000,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                           10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame,
                        cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None,
                             **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
second_mask = np.zeros_like(old_frame)
second_frame = np.zeros_like(old_frame)
x, y = 150, 150
height, width, layer = old_frame.shape
out_video = cv2.VideoWriter('../out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (width, height))
while True:
    try:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame,
                                  cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                               frame_gray,
                                               p0, None,
                                               **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        vectors = good_new - good_old
        super_vector = np.zeros(shape=(1, 2))
        for vector in vectors:
            super_vector += vector

        if vectors.shape[0] != 0:
            super_vector /= vectors.shape[0]

        second_mask = cv2.line(second_mask, (x, y), (x + int(super_vector[0][0]),
                                       y + int(super_vector[0][1])),
                               color[random.randint(1, 99)].tolist(), 2)

        frame = cv2.circle(frame, (x, y), 5,
                        color[random.randint(1, 99)].tolist(), -1)

        x, y = x - int(super_vector[0][0]), y - int(super_vector[0][0])

        img2 = cv2.add(frame, second_mask)
        img = cv2.add(frame, mask)
        out_video.write(img2)

        cv2.imshow('frame', img2)

        k = cv2.waitKey(25)

        if k == 25:
            break

        # Updating Previous frame and points
        old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1, 1, 2)
    except:
        break

out_video.release()

cv2.destroyAllWindows()
cap.release()