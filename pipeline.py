from calibration import *
from warp import *
from threshold import *
from lanes import *
import numpy as np
from moviepy.editor import VideoFileClip


def main():
    mtx, dist = find_cal_matrix()
    # use a sample image to pull transform matrix
    image = cv2.imread('test_images/test5.jpg')
    undist = undistort_img(image, mtx, dist)
    M = get_trans_mtx(undist)
    Minv = np.linalg.inv(M)

    def process(image):
        undist = undistort_img(image, mtx, dist)
        blur = cv2.bilateralFilter(undist, 25, 100, 100)
        thresh = combine_thresholds(blur, k_size_sobel=11, thresh_sobel=(20, 100),
                                    k_size_mag=9, thresh_mag=(30, 100),
                                    k_size_dir=15, thresh_dir=(0.7, 1.3))
        color_grad_thresh = combine_color_grad_thresholds(blur, thresh,
                                                          space=cv2.COLOR_RGB2HLS,
                                                          channel=2, thresh=(150, 250))
        transformed_img = warp_perspective(color_grad_thresh, M)
        return process_image_for_lanes(transformed_img, undist, Minv)

    input_video, output_video = 'project_video.mp4', 'results.mp4'
    clip1 = VideoFileClip(input_video)
    processed_clip = clip1.fl_image(process)
    processed_clip.write_videofile(output_video, audio=False)


if __name__ == "__main__":
    main()
