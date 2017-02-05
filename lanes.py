import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_window_centroids(image, window_width, window_height, margin):

    window_centroids = []
    # template window for convole
    window = np.ones(window_width)

    # initial centers

    l_center = np.argmax(np.convolve(window, np.sum(image[int(image.shape[0]*.75):, 0:int(image.shape[1]/2)], axis=0)))-window_width/2
    r_center = np.argmax(np.convolve(window, np.sum(image[int(image.shape[0]*.75):, int(image.shape[1]/2):], axis=0)))-window_width/2+int(image.shape[1]/2)
    print(r_center)
    # add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # go through each layer looking for max pixel locations
    img_h = image.shape[0]
    img_w = image.shape[1]
    for level in range(1, int(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        img_slice = image[img_h-window_height*(level+1):img_h-window_height*level, :]
        slice_sum = np.sum(img_slice, axis=0)
        slice_conv = np.convolve(window, slice_sum)
        # find the best left centroid by using past left center as a reference
        l_lower = int(max(l_center+window_width/2-margin, 0))
        l_upper = int(min(l_center+window_width/2+margin, img_w))
        l_center = np.argmax(slice_conv[l_lower:l_upper])+max(l_center-margin, 0)
        # find the best right centroid by using past right center as a reference
        r_lower = int(max(r_center+window_width/2-margin, 0))
        r_upper = int(min(r_center+window_width/2+margin, img_w))
        print(np.argmax(slice_conv[r_lower:r_upper]))
        r_center = np.argmax(slice_conv[r_lower:r_upper])+max(r_center-margin, 0)
        # add what we found for that slice
        window_centroids.append((l_center, r_center))

    return window_centroids


def fit_quad_to_meters(yvals, xvals):
    # conversion factor for meters per pixel in y dimension
    ym_per_pix = 30/720
    # conversion factor for meters per pixel in x dimension
    xm_per_pix = 3.7/700
    # convert points
    yvals *= ym_per_pix
    xvals = [val*xm_per_pix for val in xvals]
    return np.polyfit(yvals, xvals, 2)


def process_window_centers(image, window_width, window_height, margin):
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
            # print(window_centroids[level][1])
            rightx.append(window_centroids[level][1])

        return res_yvals, leftx, rightx
    else:
        raise Exception('No lane centroids found')


def fit_lane_boundaries(ymax, res_yvals, leftx, rightx):
    # fit the lane boundaries
    yvals = np.arange(0, ymax)
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals*yvals + left_fit[1]*yvals + left_fit[2]
    left_fitx = np.array(left_fitx, np.int32)

    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals*yvals + right_fit[1]*yvals + right_fit[2]
    right_fitx = np.array(right_fitx, np.int32)

    left_lane = np.array([list(zip(left_fitx, yvals))], np.int32)
    right_lane = np.array([list(zip(right_fitx, yvals))], np.int32)

    return left_lane, right_lane


def draw_lanes_on_warped(warped_img, left_lane, right_lane):
    output = np.array(cv2.merge((warped_img, warped_img, warped_img)), np.uint8)
    cv2.polylines(output, [left_lane], isClosed=False, color=[255, 0, 0], thickness=3)
    cv2.polylines(output, [right_lane], isClosed=False, color=[0, 0, 255], thickness=3)
    return output


def find_curvature(ymax, polyfit):
    ym_per_pix = 30/270
    return ((1 + (2*polyfit[0]*ymax*ym_per_pix + polyfit[1])**2)**1.5) / np.absolute(2*polyfit[0])


def process_image_for_lanes(image, window_width, window_height, margin):
    ymax = image.shape[0]
    res_yvals, leftx, rightx = process_window_centers(image, window_width, window_height, margin)
    left_lane, right_lane = fit_lane_boundaries(ymax, res_yvals, leftx, rightx)
    left_curve = find_curvature(ymax, fit_quad_to_meters(res_yvals, leftx))
    right_curve = find_curvature(ymax, fit_quad_to_meters(res_yvals, rightx))
    print(left_curve, right_curve)
    return draw_lanes_on_warped(image, left_lane, right_lane)

