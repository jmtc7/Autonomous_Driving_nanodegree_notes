# **Advanced Lane Finding Project**

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./images/01_undistortion.png "Sample undistortion"
[image2]: ./images/02_double_thresholding.png "Color and gradient combined thresholding"
[image3]: ./images/03_transformation.png "Perspective transformation/warping"
[image4]: ./images/04_polynomial.png "Sliding windows and polynomial fitting"
[image5]: ./images/05_results.png "Sample final results"


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use hybrid thresholding (color transforms and gradients), to create a binary image.
* Apply a perspective transform to rectify binary image (top-down perspective).
* Detect lane pixels and fit a quadratic polynomial to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

This work will be followed by a brief documentation/overview contained in this file. This project is a completed version of the sample project template provided by the Self-Driving Car Engineer Udemy's nanodegree. The un-completed original version is [this repository](https://github.com/udacity/CarND-Advanced-Lane-Lines).

A demo video of the performance of this approach is available in YouTube:
[![Demo video](https://www.youtube.com/watch?v=C74pNUi5B5w/0.jpg)](https://www.youtube.com/watch?v=C74pNUi5B5w)



## Camera Calibration

The followed steps in order to perform the camera calibration for this project were:

- Get a **selection of images** of a chessboard with from several X and Y positions, with different inclinations and sizes. This was provided as part of the project template repository.
- Define the **real positions** (*object points*) of the inner corners (we know how many they are, the size of the squares and that all of them are on Z=0).
- Search the **inner corners** (*image points*) in each picture (it is easy due to the high contrast of the shapes).
- Use the object and image points to calibrate the camera, getting the **camera matrix** (internal camera parameters) and **distortion coefficients** of the used lens.

This information allows us to correct the (radial or tangential) distortion in the images that may cause considerable deformations in the objects that appear in the image plane, modifying their shapes or making them seem to be farther away than what they actually are. Here is an example of the distortion correction that can be archieved by taking this into account:

![Undistortion][image1]

Note: This image was taken with the camera used to record the data for the development of this project and the image is one of the used for the camera calibration.


## Pipeline

My pipeline can be divided in the 7 steps explained during the following sub-sections. In the **iPython Notebook *P2.ipynb***, the code corresponding to all of these sections is contained in properly divided **blocks headed by an identificative title**. They also have **a lot of comments**, so I will keep this writeup brief and focused on explaining the conceps, not the implementation.

The other notebook (*P2_image_test.ipynb*) contains the pipeline for the test images, in a more *dirty* way, focused on experimentation and quick testing.


### Step 1. Image undistortion

As mentioned before,this is an esential step to make sure we are working with precise representations of our environment. We can perform it, knowing the camera matrix (*mtx*) and the distortion parameters (*dist*) using the *cv2.undistort()* function.


### Step 2. Hybrid thresholding

This is probably the most critical step. It consist in filter the pixels to choose which of them are relevant (or not useless) to solve our problem. For this project, the chosen approach was to **combine a color filter with a gradient one**, so both colors and shapes will be taken into account.

#### Step 2.1. Color filtering

In order to discriminate pixels according to their visual aspect, the first measure was to separate color and brightness substituting the conventional RGB color space by the more appropiate HLS. In particular, the S (Saturation) channel seems to provide a good and robust response representing the lane lines, so the filtering was done using only this channel.

#### Step 2.2. Gradient filtering

Regarding how the shape of the lines and its neighbourhood was taken into account, the gradients were the chosen method. This algorithm thresholds the X and Y gradients of the image, as well as their magnitude and direction. All these aspects are combined to get only one gradient-based mask that will be combined with the color one providing the results shown in the next image:

![Hybrid thresholding][image2]


### Step 3. Perspective transformation (*warping*)

Once having the relevant pixels selected, it will be way easier to work with them if we use a top-down view, which will be way less deformed and affected by perspective.

In order to get this representation, it is necessary to define 4 reference points (*source points*) from the original image and 4 points in a new image (*destination points*). This las set of points will be the positions where the source ones will be moved. The transformation needed to transform the source points into the destination ones can be used to transform the whole image so we will be able to get our bird-eye perspective of the lane.

The result of applying this warping to a lane image can be appreciated in the following picture:

![Perspective transformation][image3]


### Step 4. Lane finding

This step is, alongside with the multi-filtering one, the most critical. It consist in, given the warped binary image showing the relevant points of the Region of Interest (RoI) of the image from a top-down perspective, decide which ones of them correspond to which lane line (left or right).

This can be done in several ways, from the simple one used in the [first project](https://github.com/jmtc7/self-driving-car-nanodegree/tree/master/part1-CV-DL-and-sensor-fusion/module01-lane-finding) of the nanodegree to more complex and advanced ones such as the one used in this project. The outcome of the method used here are two quadratic polynomials, one fitted to each lane line. The method can be divided in the three sub-steps explained next.

#### Step 4.1. Histogram detection

The first thing to do is to detect where to start our search. In order to do that, we assume that the lanes in the warped picture will be close to be vertical so a good point to start searching them are the two neighbourhoods around the X coordinates that store a bigger ammount of white points accross their Y dimension.

#### Step 4.2. Sliding window

Once having an approximation of where do the lane lines start, we can generate search windows to look for pieces of them. For this, some hyperparameters will be required:

- **Number of windows** accross the Y axis.
- **Width of the windows** in the X axis.
- **Minimum number of points** that must be detected inside the window to move it from its last X position.


This can be clearly visualized in the following video (from Udacity):

[![Sliding windows video](https://www.youtube.com/watch?v=siAMDK8C_x8/0.jpg)](https://www.youtube.com/watch?v=siAMDK8C_x8)

#### Step 4.3. Polynomial fit

Last, once we know which pixels correspond to which line (which is done by the *find_lane_pixels()* in my code), we can try to fit a quadratic polynomial to the curve formed by the mean points in each window using the *np.polyfit()* function. This last part is done in my *fit_polynomial()* function.

The result of combining the lane lines pixel seach with the polynomial fitting is the following:

![Lane lines and polynomial][image4]


### Step 5. Curvidity and position estimation

Once knowing a mathematical representation of the lane limits, we can easilly compute the radius of this function at any given point, as explained [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php). However, this will give us the radius in pixels. In order to get this estimation in meters, we need to do some assumptions, such as the estimated lane length and width (30 m and 3.7 m in my code, based on the [US regulations](http://onlinemanuals.txdot.gov/txdotmanuals/rdw/horizontal_alignment.htm#BGBHGEGC), where the used videos and pictures were registered).

Regarding the car position, given that the camera is located in the center of the car, we can easilly compute the distance in pixels from both lanes to the camera and, using our assumed lane width, know which is the *real-world equivalent distance*. In the project, I have computed the distance to the middle of the lane in a way that it will be positive if the car is more on the right side and negative if it is more on the left one.


### Step 6. Inverse lane warping

Once having all the data, we can visualize it building a mask using a sampling of the polynomial equations in the warped plane and project this mask back into the original image using the inverse of the used transformation matrix. The results can be appreciated in the example image of the final results.


### Step 7. Output frame composition

Finally, in order to visualize the segmented lane, the curve radius (mean of the two lanes' radius), the relative car position and some intermedial outputs of the process, I have added some text to the main output frame as well as an small picture showing the segmented lane lines, the sliding windows and the fitted polynomials.

This layout and the performance of my implementation can be qualitativelly evaluated in the following frames:

![Sample final results][image5]



## Discussion

### Problems during the implementation

The main issue was the fine-tuning of the hyperparameters of the hybrid thresholding and of the lane finding. I also had some problems whith dimensions or type matching that I could solve using the libraries' documentation.


### Implementation weaknesses

Probably the biggest assumption done in this project is the **pixel-to-meter ratio**. If the car is not parallel to the road, the pixel-to-meter ratio will not be trustable (e.g. very steep slopes). Another obvious problem is the adaptability of the system to different country regulation regarding the lane width.

A big problem related to the **sliding windows** approach is that, if during a **very closed curve** one of the lane lines crosses the lateral limit of the warped image, the sliding windows of the top of the image will not be able to track it, which will lead to a bad polynomial fitting and, therefore, a poor overall performance. A way of fixing this is to modify the RoI according to the last computed curvature.


### Possible improvements

Instead of using a hard conventional threshold applying, it may be worth to try using a **better binarization technique**, such as adaptive gaussian thresholding or Otsu's binarization.

A measure that could help improve the **smoothness and robustness** of the approach is to use a moving average for the polynomial so that it will not be affected by punctual perturbances such as road bumps, sun flashes, line absence, etc.
 

Regarding the **optimization**, instead of doing fresh search for every new frame, it would be a good option to search wherever the windows were located in the previous frame.

