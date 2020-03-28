# Visual distortion and camera calibration
 
Robotics can be divided in 3 main parts:
- Perceive the environment using sensors
- Process the perception to decide what to do
- Perfom an action according to the decision

Regarding perception, using cameras instead of RADARs or LIDARs provide cheaper development and a really high spatial resolution, even we have to work in 2D.

The target of this module is develop an advanced lane detector able to deal with curved lanes, shadows, line color change, etc. It will also be able to tell the lane curvature or the relative car position. After, there will be a second phase in which we will add vehicle detection and tracking.


## Origin of camera distortion

In order to measure the lane curvature or the position of the car inside the lane, we will need to have a vertical perspective. For that, it is necessary to understand how data is projected by the camera from the 3D reality to the 2D image plane.

Distorsion makes the objects to be represented unacuratelly in the images (wrong size, shape or distance to them, different from different points of view, etc.).

The cameras work according to the **Pinhole model**, thanks to which a 3D point *P(X,Y,Z)* is projected to a 2D one *p(x,y)*. This is modeled by the camera matrix or de *internal camera parameters*.

However, usually, real cameras use lenses to process more information at a time, which usually introduce distortion.


## Types of camera distortion

The most comon type of distortion is the **radial distortion**, due to a curvature introduced into the light beams, which makes the objects to be more or less curved than they actually are.

Another distortion type is **tangential distortion**, caused when the camera lense is not perfectly aligned with the image plane. This makes the images to look tilted, so the objects will seem to be further or closer than they actually are.

Some lenses, such as the *fish-eye* ones, distort the images in purpuse with a certain purpose such as including more information in the image plane or just for aesthetic reasons.


## Distortion model

In order to remove the distortion from our images we need a model of it. For that, we use the **distortion coefficients**. Usually, 3 for radial distortion (k1, k2 and k3) and 2 for the tangential one (p1 and p2).

Usually k3 is close or equal to 0. It is only crucial when dealing with *major* radial distortion (i.e. wide angle lenses).


We can apply thei radial distortion coeefficients as follows:

*x\_dist = x\_real * (1 + k1*r^2 + k2*r^4 + k3*r^6)*

*y\_dist = y\_real * (1 + k1*r^2 + k2*r^4 + k3*r^6)*


And, regarding the tangential distortion:

*x_\real = x\_dist + [2*p1*x\_dist*y\_dist + p2*(r^2 + 2*x\_dist^2)]*

*y_\real = y\_dist + [p1*(r^2 + 2*y\_dist^2) + 2*p2*x\_dist*y\_dist]*


## Camera calibration

In order to get the camera intrinsic parameters and the distortion coefficients, we need objects of known shapes to stablish how they are projected in our image plane. The most common object used for this purpose is a chessboard from which we know the ammount of squares and their size. Due to the high contrast of the shapes, it is also really easy to detect the vertices. 

In the script *draw_vertices.py*, the ammount of X and Y inner vertices is configured and I use two OpenCV functions to detect and draw the chessboard vertices.


For a full calibration, it is recommended to use at least 20 images of the chessboard taken from different angles and distances (X and Y axis, closer and further and with more and less inclination). We can also add one or more additional test images not used for the calibration to test our result.

We can create a list of the 3D real points (*object points*) and another one with their location after being projected with the camera and lens (*image points*). We can use the *cv2.calibrateCamera()* function to get the camera parameters (*mtx*) and distortion coefficients (*dist*) using these points. The *object point* of the top left corner will be (0, 0, 0), all of them will keep the z=0 and the X and Y coordinates will be obtained using the amount of squares and their size.

To undistort a new image, we can use *cv2.undistort()*, which uses the camera matris (*mtx*) and the distortion coefficients of the used lens (*dist*).


All this is done in the *undistort.py* script. The object and image points contained in the pickle file are extracted from the images in [this repository](https://github.com/udacity/CarND-Camera-Calibration). The pickle file itself has been downloaded from [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Advanced_Lane_Finding_Images/correct_for_distortion/wide_dist_pickle.p) and it contains the object and image points, the camera matrix and the distortion coefficients, but these two have been computed in the script for practice.


---

# Lane curvatue

We turn the steering wheel more or less depending on how much the lane is curved. The ammount of necessary turn can be computed if we know the current position of the car in the lane, the lane curvature (desired future car position) and speed and dynamics of the car.

To get the lane curvature, we will start by detect which pixels correspond to the lane limits (lane detector project). After that, we will transform the image to get a bird eye view of the lane. Next, we can fit a polynomial equation to the lane lines (that is why we need the bird eye view). Finally, we can get the lane curvature from this polynomial.


## Perspective transform

Due tu perspective, the farther an object is, the smaller it is represented. Furthermore, parallel lines converge to a point. To tranform the perspective of an image, we modify the *real* Z coordinate of the 3D point that originated each pixel to make 2D points to seem closer or farther away from the camera.

Getting top-down views of roads can help us localize in maps, since they use this perspective.

To transform the perspective of an image, we will use a similar process as for undistorting images, but instead of projecting object points to image points, we will project image points into a certain desired locations inside the image.


## Traffic sign transform

We will start by defining a rectangle defining a plane in the traffic sign (four points are enaugh to make a perspective tranform). Afterwards, we need to choose where we want these 4 points to appear in our output image with the transformed perspective (they must be in a straight rectangle so we can get a front view of the sign).

The transformation between the two sets of points is done by *cv2.getPerspectiveTransform()* and the transformation itself is performed by *cv2.warpPerspective()*. This last function will require a size for the output image (we can use the one of the input image) and an interpolation method to fill the pixels for which we will not have data (linear interpolation (cv2.INTER\_LIENEAR) can do the job).


## Sample perspective transform

A sample perspective transform is implemented in the *perspective_transformation.py* script. In it, a distorted image is taken as an input and it is undistorted using the camera matrix and distortion coefficients. After that, it is transformed to get a top-down view of the sample chessboard used.


