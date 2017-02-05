import cv2
import numpy as np


# src coordinates
src = np.float32([
    [570, 460],
    [155, 705],
    [1200, 705],
    [740, 460]
])

# dest coordinates
dst = np.float32([
    [175, 20],
    [155, 720],
    [870, 720],
    [1000, 20]
])


# Note image is undistorted image
def get_trans_mtx(image):
    img_size = (image.shape[1], image.shape[0])
    trans_mtx = cv2.getPerspectiveTransform(src, dst)
    return trans_mtx


def warp_perspective(image, M):
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
