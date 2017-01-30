import numpy as np
import matplotlib.pyplot as plt


# window settings
window_width = 100
window_height = 40
margin = 100


def find_window_centroids(image, window_width=window_width, window_height=window_height, margin=margin):

    window_centroids = []
    # template window for convole
    window = np.ones(window_width)

    # initial centers

    l_center = np.argmax(np.convolve(window, np.sum(image[int(image.shape[0]*.75):, 0:int(image.shape[1]/2)], axis=0)))-window_width/2
    r_center = np.argmax(np.convolve(window, np.sum(image[int(image.shape[0]*.75):, int(image.shape[1]/2):], axis=0)))-window_width/2+int(image.shape[1]/2)

    # add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # go through each layer looking for max pixel locations
    img_h = image.shape[0]
    img_w = image.shape[1]
    for level in range(1, int(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        slice = image[img_h-window_height*(level+1):img_h-window_height*level, :]
        slice_sum = np.sum(slice, axis=0)
        slice_conv = np.convolve(window, slice_sum)
        # find the best left centroid by using past left center as a reference
        l_lower = int(max(l_center+window_width/2-margin, 0))
        l_upper = int(min(l_center+window_width/2+margin, img_w))
        l_center = np.argmax(slice_conv[l_lower:l_upper])+max(l_center-margin, 0)
        # find the best right centroid by using past right center as a reference
        r_lower = int(max(r_center+window_width/2-margin, 0))
        r_upper = int(min(r_center+window_width/2+margin, img_w))
        r_center = np.argmax(slice_conv[r_lower:r_upper])+max(r_center-margin, 0)
        # add what we found for that slice
        window_centroids.append((l_center, r_center))

    return window_centroids


def process_window_centers(image, window_width=window_width, window_height=window_height, margin=margin):
    window_centroids = find_window_centroids(image, window_width, window_height, margin)

    if len(window_centroids) > 0:

        # points used to draw the full left and right lanes, and windows
        rightx = []
        leftx = []

        # each centroid ypoint per layer
        res_yvals = np.arange(image.shape[0]-window_height/2, 0, -window_height)

        # go through each level and store left and right centers
        for level in range(0, len(window_centroids)):
            leftx.append(window_centroids[level][0])
            rightx.append(window_centroids[level][1])

        # fit the lane boundaries
        yvals = np.arange(0, image.shape[0])
        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
        left_fitx = np.array(left_fitx, np.int32)

        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
        right_fitx = np.array(right_fitx, np.int32)

        left_lane = np.array([list(zip(left_fitx, yvals))], np.int32)
        right_lane = np.array([list(zip(right_fitx, yvals))], np.int32)

        return left_lane, right_lane
    else:
        raise Exception('No lane centroids found')


def draw_lanes_on_warped(warped_img, left_pts, right_pts):
    output = np.array(cv2.merge((warped_img, warped_img, warped_img)), np.uint8)
    cv2.polylines(output, [left_lane], isClosed=False, color=[255, 0, 0], thickness=3)
    cv2.polylines(output, [right_lane], isClosed=False, color=[0, 0, 255], thickness=3)
    return ouput


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
