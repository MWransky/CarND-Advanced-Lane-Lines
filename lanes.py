import numpy as np
import matplotlib.pyplot as plt
import cv2


def find_histogram_peaks(warped, side):
    histogram = np.sum(warped[warped.shape[0]/2:, :], axis=0)
    # Find the peak of the left and right
    midpoint = np.int(histogram.shape[0]/2)
    if side == 'left':
        return np.argmax(histogram[:midpoint])
    else:
        return np.argmax(histogram[midpoint:]) + midpoint


def find_line_pts_blind(warped, peak, line):
    # Number of sliding windows
    nwindows = 13
    # Set height of windows
    window_height = np.int(warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    current = peak
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_x_low = current - margin
        win_x_high = current + margin
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            current = np.int(np.mean(nonzerox[good_inds]))

    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)

    # Extract line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    if len(x) == 0:
        line.detected = False
        lane_inds = line.indx
        x = line.allx
        y = line.ally
    else:
        line.detected = True

    # Fit a second order polynomial to each
    fit = np.polyfit(y, x, 2)

    fit_meters = fit_quad_to_meters(y, x)

    return fit, fit_meters, lane_inds, x, y, line


def find_line_pts(warped, fit, line):
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    lane_inds = ((nonzerox > (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0]*(nonzeroy**2) + fit[1]*nonzeroy + fit[2] + margin)))

    # Extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]
    if len(x) == 0:
        line.detected = False
        lane_inds = line.indx
        x = line.allx
        y = line.ally

    fit = np.polyfit(y, x, 2)

    fit_meters = fit_quad_to_meters(y, x)

    return fit, fit_meters, lane_inds, x, y, line


def visualize_lanes(warped, left_fit, right_fit, left_lane_inds, right_lane_inds):
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    out_img = np.dstack((warped, warped, warped))*255

    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.show()
    return


def find_window_centroids(image, window_width, window_height, margin):

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
        img_slice = image[img_h-window_height*(level+1):img_h-window_height*level, :]
        slice_sum = np.sum(img_slice, axis=0)
        slice_conv = np.convolve(window, slice_sum)
        # find the best left centroid by using past left center as a reference
        l_lower = int(max(l_center+window_width/2-margin, 0))
        l_upper = int(min(l_center+window_width/2+margin, img_w))
        if np.argmax(slice_conv[l_lower:l_upper]) == 0:
            l_center = max(l_center-margin/4, 0)
        else:
            l_center = np.argmax(slice_conv[l_lower:l_upper])+max(l_center-margin, 0)
        # find the best right centroid by using past right center as a reference
        r_lower = int(max(r_center+window_width/2-margin, 0))
        r_upper = int(min(r_center+window_width/2+margin, img_w))
        if np.argmax(slice_conv[r_lower:r_upper]) == 0:
            r_center = max(r_center-margin/4, 0)
        else:
            r_center = np.argmax(slice_conv[r_lower:r_upper])+max(r_center-margin, 0)
        # add what we found for that slice
        window_centroids.append((l_center, r_center))

    return window_centroids


def fit_quad_to_meters(yvals, xvals):
    yvals = [float(yval) for yval in yvals]
    xvals = [float(xval) for xval in xvals]
    # conversion factor for meters per pixel in y dimension
    ym_per_pix = 30/720
    # conversion factor for meters per pixel in x dimension
    xm_per_pix = 3.7/700
    # convert points
    yvals = [val*ym_per_pix for val in yvals]
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


def draw_lanes_on_original(warped, undist, leftLine, rightLine, Minv):
    left_fit = leftLine.best_fit
    right_fit = rightLine.best_fit

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    # Add text to dispaly lane curvatures
    curve = (leftLine.radius_of_curvature + rightLine.radius_of_curvature)/float(2)
    ymax = warped.shape[0]
    img_center = warped.shape[1]/float(2)
    off_center = find_center(ymax, img_center, left_fit, right_fit)
    text = 'Average curvature: {0} m; Off-center by {1} m'. format(round(curve), round(off_center, 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    result = cv2.putText(result, text, (100, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return result


def find_curvature(ymax, polyfit):
    ym_per_pix = 30/270
    return ((1 + (2*polyfit[0]*ymax*ym_per_pix + polyfit[1])**2)**1.5) / np.absolute(2*polyfit[0])


def find_center(ymax, img_center, left_fit, right_fit):
    xm_per_pix = 3.7/700
    img_center *= xm_per_pix
    left_pt = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    right_pt = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]

    lane_cntr = (left_pt*xm_per_pix + right_pt*xm_per_pix)/float(2)
    return abs(img_center - lane_cntr)


def process_line(warped, line, side):
    ymax = warped.shape[0]

    if not line.detected:
        # Lane line is not detected, find it
        # Next proceed with line finding algorithm
        peak = find_histogram_peaks(warped, side)
        fit, fit_m, indx, x, y, line = find_line_pts_blind(warped, peak, line)
        curve = find_curvature(ymax, fit_m)
        if line.should_update_blind(curve, fit):
            line.update(fit, indx, x, y)
            line.detected = True
    else:
        # Lane line detected in last frame, use as reference
        prev_fit = line.best_fit
        fit, fit_m, indx, x, y, line = find_line_pts(warped, prev_fit, line)
        curve = find_curvature(ymax, fit_m)
        if line.should_update(curve, fit):
            line.update(fit, indx, x, y)
        else:
            line.detected = False
            # if line.grace_amount >= 5:
            #     # unable to find lane line 5 frames in a row, start fresh
            #     line.grace_amount = 0
            #     line.detected = False

    return line


def process_image_for_lanes(warped, undist, Minv, leftLine, rightLine):
    leftLine = process_line(warped, leftLine, 'left')
    rightLine = process_line(warped, rightLine, 'right')
    # visualize_lanes(warped, left_fit, right_fit, l_indx, r_indx)

    output = draw_lanes_on_original(warped, undist, leftLine, rightLine, Minv)
    return output, leftLine, rightLine
