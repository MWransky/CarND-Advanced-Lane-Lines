import cv2
import numpy as np

top_right_src = [720, 460]
top_left_src = [570, 460]
bottom_right_src = [1200, 705]
bottom_left_src = [155, 705]

top_right_dest = [870, 100]
top_left_dest = [175, 100]
bottom_right_dest = [870, 705]
bottom_left_dest = [175, 705]
# src coordinates
src = np.float32([
    top_right_src,
    bottom_right_src,
    bottom_left_src,
    top_left_src
])

# dest coordinates
dst = np.float32([
    top_right_dest,
    bottom_right_dest,
    bottom_left_dest,
    top_left_dest
])


# Note image is undistorted image
def get_trans_mtx(image):
    img_size = (image.shape[1], image.shape[0])
    trans_mtx = cv2.getPerspectiveTransform(src, dst)
    return trans_mtx


def warp_perspective(image, M):
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
