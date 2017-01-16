import numpy as np
import matplotlib.pyplot as plt


def find_peaks(mask):
    # find a histogram for values along bottom half of left image
    hist_left = np.sum(mask[int(mask.shape[0]/2):, 0:mask.shape[1]/2], axis=0)
    # find a histogram for values along bottom half of right image
    hist_right = np.sum(mask[int(mask.shape[0]/2):, mask.shape[1]/2:], axis=0)
    # find index for highest value in histograms
    idx_left = hist_left.argsort()[-1:][::-1]
    # add half length of image to right index
    idx_right = hist_right.argsort()[-1:][::-1] + mask.shape[1]/2

    pt1 = idx_left[0]
    pt2 = int(idx_right[0])
    return pt1, pt2


def find_lane_pts(mask, pt, lane_width=100):
    left_limit = pt - int(lane_width/2)
    right_limit = pt + int(lane_width/2)
    roi = mask[:, left_limit:right_limit]
    nonzero_pts = np.nonzero(roi)
    y = nonzero_pts[0] + int(mask.shape[0]/2)
    x = nonzero_pts[1] + left_limit
    return y, x


def fit_function(y, x):
    return np.polyfit(y, x, 2)


def quadratic(params, y):
    return params[0]*y**2 + params[1]*y + params[2]


def create_rgb_image(mask):
    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = mask * 255
    return img


def draw_lines_on_img(img, params):
    inputs = np.arange(img.shape[0])
    outputs = quadratic(params, inputs)
    for i in range(outputs.shape[0]):
        x = int(outputs[i])
        y = int(inputs[i])
        if x > img.shape[1]:
            x = img.shape[1] - 1
        if x < 0:
            x = 0
        img[y, x, 2] = 255

    return img


def process_lane_finding(mask):
    pt1, pt2 = find_peaks(mask)
    laney_left, lanex_left = find_lane_pts(mask, pt1)
    laney_right, lanex_right = find_lane_pts(mask, pt2)

    params_left = fit_function(laney_left, lanex_left)
    params_right = fit_function(laney_right, lanex_right)

    img = create_rgb_image(mask)

    img = draw_lines_on_img(img, params_left)
    img = draw_lines_on_img(img, params_right)

    return img
