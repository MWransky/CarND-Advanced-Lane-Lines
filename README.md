# Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:  

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply the distortion correction to the raw image.  
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view"). 
* Detect lane pixels and fit to find lane boundary.
* Determine curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

---
## Processing Pipeline
### Camera Calibration
Funtions for determining the distortion coefficients for the camera as well as the method to use these coefficients to undistort an image are in the `calibration.py` file. Methods for this calibration process utilize OpenCV's `cameraCalibration` function as discussed in their [docs](http://docs.opencv.org/3.1.0/dc/dbb/tutorial_py_calibration.html).

Here is an example image used to determine the calibration coefficients both before and after correction.

Before:
![dist_checker](https://cloud.githubusercontent.com/assets/13735131/23767950/67a06518-04d8-11e7-88b1-933a3dee6b98.png)

After:
![undist_checker](https://cloud.githubusercontent.com/assets/13735131/23767954/67a365d8-04d8-11e7-9163-b24d4a3f7134.png)

### Distortion Correction
Using the coefficients learned via the test calibration images, we take each image in the pipeline and undistort the image. See below for a distorted image and the resulting corrected image.

Distorted:
![original](https://cloud.githubusercontent.com/assets/13735131/23767958/67b606fc-04d8-11e7-8160-8842eadc9cb4.png)

Corrected:
![undistorted](https://cloud.githubusercontent.com/assets/13735131/23767957/67b14e14-04d8-11e7-8c88-378b2352e18a.png)

### Thresholding
In order to interpret features in the image we combine three main thresholding techniques:
* Sobel (Based on direction of gradient)
* Sobel (Based on magnitude of gradient)
* Color

All thresholding methods can be found in the `threshold.py` file. Each thresholding method generates a binary mask and then these masks are compared and combined to produce the final output. Specific parameters for each method were tuned and tested to produce the best results in a variety of road conditions. These specific parameters can be found in the implementation of the thresholding functions in the `pipeline.py` file.

Here is an example binarized image produced by the thresholding process.
![thresholded](https://cloud.githubusercontent.com/assets/13735131/23767951/67a0e808-04d8-11e7-9479-e8d1f2fafe1b.png)

### Perspective Transform
Because we need a consistant perspective to determine the lane directions, we must apply a perspective transform to take the information from the original orientation to a "bird's eye view" perspective. Code responsible for this process is under `warp.py` and the functions utilize OpenCV's geometric transformation functionalities as discussed [here](http://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html).

Here is an example output of the perspective transform for the binarized mask.
![warped](https://cloud.githubusercontent.com/assets/13735131/23767953/67a1faea-04d8-11e7-8bc1-246a06d47263.png)

### Finding the Lanes
Once we have the binarized, perspective transformed information we then proceed to search for the left and right lane lines. First, we take a segment of the image, convert the binary information into a histogram of accumulated values in the vertical, and find the peak of this histogram. This peak is the seed location for our lane line. Next, we take this seed location and find the remaining lane line pixels by growing the points window slice by window slice of the image. Finally, these points are feed into a polynomial approximation that produces the polynomial coefficients for the quadratic line that best fits the points.

All code for each of these steps can be found in the `lanes.py` file. See below for an example of the polynomial fits plotted on the binarized mask.

![lines_drawn](https://cloud.githubusercontent.com/assets/13735131/23767955/67a7e7f2-04d8-11e7-9292-07b8567f447d.png)

### Keeping Track of Lanes
To simplify record keeping for all the parameters associated with each lane line throughout processing we use a `Line` class as defined in the `line.py` file. This class has objects that store information regarding the current fit and methods to determine if updates to parameters should be made. Additionally, this line class keeps track of the latest 5 polynomial fits and averages them. If the average exists, the lane finding algorithm can start its search for lane pixels from the last known locations. However, if the lane cannot be found from the last known location, or the lanes we find fail to be within a reasonable curvature (1500 m or less) we start the search again blind and erase the average fit information. Each lane line has its separate `Line` class instance to simplify logic.

### Curvature and Car Placement
Before we can use this information and translate it to real-world insights we must convert from pixel space to metric space. There are conversion factors that make this process simple as demonstrated in the `fit_quad_to_meters` function within `lanes.py`. This function takes the pixel space polynomial fit and finds the fit for the lines in metric space.

We find two main measurements from our lane information:
* Radius of curvature:
  * We find the radius of curvature using the coefficients from the metric fit of the line and its resulting derivative. The result is calculated in the `find_curvature` and uses the maximum y-value (max image height) as the input for the curvature equation. We average the curvature based on both the left line and right line.
* Car placement:
  * We also determine the placement of the car relative to the center of the found lanes. We assume that the image center is the center of the car and then we find the middle value between the left lane line and the right lane line at the maximum y-value (basically the location closest to the car). Finally, we find the difference between these two values and that represents the distance off-center from the lane. The `find_center` function in `lanes.py` is responsible for this logic.
### Final Output
We combine all the lane, curvature, and car placement information and impose it directly on the image. We paint in the full lane as determined by the pixels in between the two lanes and we put text on the image for the curvature and distance from lane center. Code for this process is in the `draw_lanes_on_original` function within the `lanes.py` file.

Here is an example final output image.
![final](https://cloud.githubusercontent.com/assets/13735131/23767952/67a1336c-04d8-11e7-9af8-084cb1d99579.png)

## Video of Algorithm Performance
A video demonstrationg the algorithm's ability to identify lanes can be found [here](https://youtu.be/R1MC6SvFMhY).

## Obstacles and Future Considerations
The current algorithm successfully finds both straight lanes and curved lanes as well as lanes obscured by shadow. However, this algorithm is still unstable given harder lighting conditions (bright sunlight, raining, etc.). This is likely due to inconsistencies in the intensity values of the images that represent color in the environment. Also, image features may be hidden due to lens flare, over exposure, etc. that will cause our algorithm to fail.

Developing an algorithm to account for these more difficult environments will likely head in the direction of a neural network to identify lane placement based on an input image. Although this approach will need training data labeled across a variety of situations, the main advantage is that we will not need to hand-craft features and parameters to do the processing. Instead, the network will determine its own features to find the lanes.
