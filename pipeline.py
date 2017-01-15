from calibration import *
from warp import *
from threshold import *


def main():
    image = cv2.imread('test_images/test3.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mtx, dist = find_cal_matrix()
    undist = undistort_img(image, mtx, dist)
    thresh = combine_thresholds(undist, k_size_sobel=7, thresh_sobel=(30, 160),
                                k_size_mag=13, thresh_mag=(50, 120),
                                k_size_dir=7, thresh_dir=(50, 120))
    color_grad_thresh = combine_color_grad_thresholds(undist, thresh,
                                                      space=cv2.COLOR_RGB2HLS,
                                                      channel=2, thresh=(30, 120))
    plt.figure(1)
    plt.imshow(color_grad_thresh)
    transformed_img = warp_perspective(color_grad_thresh)
    plt.figure(2)
    plt.imshow(transformed_img)
    plt.show()

if __name__ == "__main__":
    main()
