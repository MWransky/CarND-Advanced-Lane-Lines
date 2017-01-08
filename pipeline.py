from calibration import *
from warp import *


def main():
    image = cv2.imread('test_images/test3.jpg')
    mtx, dist = find_cal_matrix()
    undist = undistort_img(image, mtx, dist)
    plt.figure(1)
    plt.imshow(undist)
    transformed_img = warp_perspective(undist)
    plt.figure(2)
    plt.imshow(transformed_img)
    plt.show()

if __name__ == "__main__":
    main()
