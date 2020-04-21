# Advanced CV Techniques

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In order to apply all of the things learned so far to a real project, these are the steps to follow:

- Calibrate the camera.
- Correct the distortion introduced by the used lens.
- Threshold the image using color and gradients.
- Transform the image perspective to get a top-down view of the lane.


## Finding the Lane Lines
Once having the top-down view, we need to find the lane lines in it. For such purpose, there are several approaches more sofisticated than the one used in the first project (Canny edge detection + Hough line creation + average lines post-processing). Some options for improving our results are the ones exposed in the following subsections.

### Histogram Peaks
With the top-down view of the mask, we can count the ammount of white pixels in each x coordinate of the bottom half of the mask (because there is where the lines will be more vertical). We will then be able to search in the generated vector (histogram) for the highest peaks, which should correspond to the lane lines. This is done in *01_histogram_peaks.py*.

### Sliding Window
An improved version of the last method is to use a sliding window that will detect histogram peaks in several regions of the Y axis. this way, instead of having only two peaks for all the image, we will have a collection of pairs of peaks, which will make our method robuts to curves. A demo of this idea can be seen [here](https://youtu.be/siAMDK8C_x8).

After this, we can fit a polynomial to properly detect the lane lines instead of having a collection of discrete samples of it.

In order to implement this, we will need to set a series of **hyperparameters**, such as:

- **Number of windows**: Ammount of slides we will use of the Y axis.
- **Margin**: Allowed variation for the positions considered as the right and left lane lines samples.
- **Minimum pixels** required to recenter the window in the current highest right and left lane lines samples.

This is implemented in the *02_sliding_window.py* script.

### Search From Prior
This is starting to become a heavy algorithm and repeating it for each new frame can become costly. Since the lane lines do not usually change a lot between two consecutive frames, we can assume that they will be close to where they were in the last frame. This allows us to search only around the lane lines detected in the previous frame, which is really similar to use a custom RoI for each frame. This strategy make our algorithm robust to sharp curves or challenging visual conditions. If at some point we lose track of the lane lines, we can always rerun the whole pipeline.

This is implemented in the *03_prior.py* script.

The **problem of the current implementation** is that, when processing images with large curves, one of the lane lines can cross one of the image limits. In this case, the window will be stuck on its last position because there will not be enough white pixels to recenter the window.


## Measuring Curvature
Once having the polynomicals of both lane lines, we can estimate the curve radius as explained [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). The script *04_curve_radius_sim.py* implements this with some artificially generated data, while another one including a tranformation from the pixel space to real world units (m) is implemented in *05_curve_radius_real.py*. In this last one, it is assumed that the lane is about 30 meters long and 3.7 meters wide. Regarding the image, our camera image has 720 relevant pixels in the y-dimension (remembering the perspective-transformation), and around 700 relevant pixels in the x-dimension.


## Further Reading
- Lane Finding with Semantic Segmentation
  - [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211) by E. Shelhamer, J. Long and T. Darrell. This technique can be used with the [KITTI road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) to build a model that can detect free space in the road.
  - [Lane Detection with Deep Learning (Part 1)](https://towardsdatascience.com/lane-detection-with-deep-learning-part-1-9e096f3320b7) and [Part 2](https://towardsdatascience.com/lane-detection-with-deep-learning-part-2-3ba559b5c5af), by M. Virgo.
- Other Lane Finding Techniques
  - [VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition](https://arxiv.org/abs/1710.06288), by S. Lee, et. al. 
- Vehicle Detection
  - [Learning to Map Vehicles into Bird's Eye View](https://arxiv.org/abs/1706.08442), by A. Palazzi, et. al. 

