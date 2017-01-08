import cv2
import numpy as np

# Define source points taken from image of straight lanes
src_pts = np.float32([[265, 720], [1160, 720], [755, 480], [582, 482]])


# Note image is undistorted image
def warp_perspective(image):
    img_size = (image.shape[1], image.shape[0])
    dst_pts = np.float32([[265, 720], [1160, 720],
                         [1160, 0], [265, 0]])
    print(src_pts, dst_pts)
    trans_mtx = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, trans_mtx, img_size)
