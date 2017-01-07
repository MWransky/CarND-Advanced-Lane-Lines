import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

# Calibration images folder
img_folder = 'camera_cal'

# Define grid corner constants
number_x = 9
number_y = 6

# Define constant object points (0, 0, 0) ... (8, 5, 0) etc
grid_points = np.zeros((number_y*number_x, 3), np.float32)
grid_points[:, :2] = np.mgrid[0:number_x, 0:number_y].T.reshape(-1, 2)


# Function to find corner points in cal images
def point_finder(grid_points=grid_points, image_folder=img_folder):
    obj_pts = []
    img_pts = []

    img_list = glob.glob('{}/calibration*.jpg'.format(image_folder))

    for index, fname in enumerate(img_list):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (number_x, number_y), None)

        if ret is True:
            obj_pts.append(grid_points)
            img_pts.append(corners)
            # To verify corners are properly found uncomment below code to view images w/ corners marked

            # cv2.drawChessboardCorners(img, (number_x, number_y), corners, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    return obj_pts, img_pts


def find_cal_matrix(obj_pts, img_pts, img_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)
    return mtx, dist


def undistort_img(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


def main():
    obj_pts, img_pts = point_finder()
    image = cv2.imread('camera_cal/calibration4.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mtx, dist = find_cal_matrix(obj_pts, img_pts, image.shape[::-1])
    undist = undistort_img(image, mtx, dist)
    plt.imshow(undist)
    plt.show()

if __name__ == "__main__":
    main()
