from calibration import *
from warp import *
from threshold import *
from lanes import *
import numpy as np


# window settings
window_width = 100
window_height = 100
margin = 20


def main():
    image = cv2.imread('test_images/test5.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mtx, dist = find_cal_matrix()
    undist = undistort_img(image, mtx, dist)
    blur = cv2.bilateralFilter(undist, 25, 100, 100)
    M = get_trans_mtx(undist)
    Minv = np.linalg.inv(M)
    thresh = combine_thresholds(blur, k_size_sobel=11, thresh_sobel=(20, 100),
                                k_size_mag=9, thresh_mag=(30, 100),
                                k_size_dir=15, thresh_dir=(0.7, 1.3))
    color_grad_thresh = combine_color_grad_thresholds(blur, thresh,
                                                      space=cv2.COLOR_RGB2HLS,
                                                      channel=2, thresh=(150, 250))
    transformed_img = warp_perspective(color_grad_thresh, M)
    img = transformed_img
    # img = cv2.fastNlMeansDenoising(transformed_img, None, 10, 7, 21)
    plt.figure(1)
    warped = warp_perspective(blur, M)
    plt.imshow(warped)
    output = process_image_for_lanes(img, undist, Minv)
    plt.figure(2)
    plt.imshow(output)
    plt.show()

if __name__ == "__main__":
    main()
