# Extended Kalman Filters (EKFs)

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


So far, we have seen how to apply KFs to a sequence of measurements of one sensor. However, we can use it to combine measurements from different sensors, such as LIDAR and RADAR and combine their strengths and comopensate eachother's weaknesses. The final target will be to track a pedestrian (its location, heading and speed) using a KF-based algorithm implemented in C++.

We will use LIDAR and RADAR data to estimate the pose of a moving pedestrian, which will be represented by a 2D position and velocity. On the first measurement we will initialize the matrices and for every other new measurement, we will perform the prediction step followed by the measurement update. The steps that will be followed are:
- We get the **first measurement** (from the RADAR or the LIDAR) of the pedestrian's position.
- **Initialize** the state and covariance matrices of the pedestrian's pose.
- We get a **second measurement** after *delta_t* time.
- **Predict** where the pedestrian will be after *delta_t* time. To do this, we can assume constant velocity.
- **Update** the pedestrian's state and covariance comparing the second measurement and the prediction using a Kalman Filter (or a modification of it).
- We will get **new measurements** that will be followed by the prediction and update steps again and again.


## Kalman Filters for Data Fusion
When we have more than one sensor, such as if we have LIDAR and RADAR, each sensor will have its own prediction/update scheme. Therefore, the belief on the pedestrian's position and velocity will be upated asynchronously. i.e. one sensor may provide an observation on time t-1, with which we will predict the time t pose. Next, on time t, the other sensor may provide the next measurement, which we will use to correct the predicted state. We will keep this loop going using the new measurements to correct our predictions regardless their source (i.e. we may need to take it into account for the implementation, but we will not be waiting for alternate readings).

In particular, while LIDARs provide measurements in cartesian coordinate systems, RADARs do so in polar ones. Thus, we will need different measurement update functions.

In case we receive data silmuntaneously from both sensors in a given time, we will execute the same loop first with one sensor and then with the other one. Theoretically, though, given that the *delta_t* has been 0, we could skipt the second prediction step because the state of the target will be the same.

In the *01_kf.cpp* code, a 1D Kalman Filter is implemented in C++.


## Multi-Dimensional Data Fusion
So far, we have implemented a 1-dimensional filter after analyzing its equations. If we want to track a pedestrian in a 2D plane using the data from our sensors, we will need more dimensions. Now, the state of the pedestrian will be formed by two positions and two velocities: *x = [p_x, p_y, v_x, v_y]*.

Now, even before we should have also done it, we will consider noise. There are two types of noise:
- **Process noise**: Since we asume constant velocity and direction, if the target changes some of this, we will have it model it with an uncertainty increase. This noise will be modeled by a 0-centered normal distribution with covariance *Q*.
- **Measurement noise**: Associated to the uncertainty about the sensor measurements.

In the *02_2D_KF* directory, some files implementing this are contained. These are three classes, the Kalman Filter one (KF code, including predic and update functions), the tracking one (will use the KF object) and the measurement package. The challenge was to modify the F and Q matrices according to the elapsed time between the current and previous measurements inside the *ProcessMeasurement()* function of the *tracking.cpp* file. After that, the code should call the *predict* and *update* KF functions. The main innovation of this challenge is that we will be processign one measurement at a time, which will be processed as *packages*. This packages will contain the raw measurement, a timestamp of the measure and the type of the sensor, but we will be using only LIDAR.


## Extended Kalman Filter (EKF)
The point on fusing LIDAR and RADAR is that, while LIDAR provides us with the position of the targets (we can process the input pointcloud to get this position), it can not give us the velocity of it, whereas RADARs can using the Doppler Effect. Having different observation matrices will allow us to reduce much more our uncertainty because we can observe different aspects of the target with different sensors.

The state transition matrix (F) will be the same. The measurements provided by the RADAR, though, will be, instead of the target position (*p_x* and *p_y*), as the (processed) LIDAR's output, it will give us *rho* (distance form the car), *phi* (angle between the direction of the car and the target) and *rho dot* (target velocity). This measurements will also be affected by a 0-mean noise, modeled by the *R* measurement covariance matrix. 

Given this new measurement vector (*z = [rho, phi, rho dot]*), we will need a differnet measurement function (*h(x)*) our state vector to this new measurement vector. This will be a non-linear function, so we will not be able to apply the Kalman Filter as we were doing it before because we will be applying a non-linear function to a gaussian, what will make it stop being a Gaussian.

NOTE: In the LIDAR case, we had *z = [p_x, p_y]*, so our measurment matrix (*H*) was a linear function that allowed us to match the state vector with the measurements with linear relationships.


In order to solve this problem with the linearity, there are several things we could do. One of them is to linearize the *h(x)* function on the point in which we are evaluating it. This is the key idea of the EKF. We will use a linear function tangent to our *h(x)* in the mean of our gaussian distribution (target pose).

For this linearization, the EKF uses the **First Order Taylor Expansion** when the measurement (*h(x)*) or the state transition (*f(x)*) are non linear. This consist in using the first order (linear) taylor aproximation of the non-linear function at the mean of the gaussian that is being used. Given that we will be using more than one variable, we will need a **multi-dimensional taylor expansion**. For that, we sould be using the Jacobian and Hessian matrices, which are matrices containing the first and second order partial derivatives of a multi-dimensional equation (respectively), and even other higher order derivatives. However, since we want a linear aproximation, we will only use the Jacobian (first order derivative).

Given that our measurements (z) have 3 dimensions (*[rho, phi, rho dot]*) and our states (x), 4 (*[p_x, p_y, v_x, v_y]*), the Jacobian matrix (*H_j*) will be 3x4. In the code of *03_jacobian.cpp*, the jacobian of h(x) is computed.


In order to add these things to our current KF implementation to transoform it to an EKF, we will use our non-linear function *f(x)* to predict the state (x') and *h(x)* to compute the measurement error (y). We will then substitute the state transition matrix *F* and the measurement matrix *H* will be substituted with the corresponding jacobians *F_j* and *H_j* (when computing P', S, K and P). By doing this last thing, we linearize our functions. NOTE: This has to be done for each point in order to linearize our functions in a way that the most similar point of the linearization is located in the mean that is being processed.


NOTE: For this problem, we have a linear motion model (*f(x)*, of the pedestrian), but chances are we will not have it if we use the KF/EKF to track the position of a robot. The state-measurement mapping (*h(x)*) is non-linear because of the relationship between the RADAR's output and the state is non-linear, which is likely to happen always.


## (E)KFs' performance evaluation
There are many ways of doing it, but one of the most common ones is the Root Mean Square Error (RMSE). It needs the *estimated state* (the pose estimation of our KF) and the *true state* (groundtuth/real position). The difference between these two values, is known as *the residual*. We will square all the residuals and compute the average (or mean) of all the squared residuals (or errors). Finally, we will compute the square root of the mean square error. The lower is the error, the more precise our estimations are.

The RMSE is implemented in the *04_rmse.cpp* code.


## Lesson summary
The idea is to fuse data from LIDAR and RADAR to estimate pedestrian's position and speed using a 2D Kalman Filter and linearization techniques. Next, we will be using Uscented Kalman Filters (UKFs) and use non-linear models for motion, not only for measurements.

Some recommended further reading is listed now:
- Tracking Multiple Objects and Sensor Fusion:
	- [No Blind Spots: Full-Surround Multi-Object Tracking for Autonomous Vehicles using Cameras & LiDARs ](https://arxiv.org/pdf/1802.08755.pdf), by A. Rangesh and M. Trivedi
	- [Multiple Sensor Fusion and Classification for Moving Object Detection and Tracking](https://hal.archives-ouvertes.fr/hal-01241846/document), by R.O. Chavez-Garcia and O. Aycard
- Stereo cameras:
	- [Robust 3-D Motion Tracking from Stereo Images: A Model-less Method](http://www.cse.cuhk.edu.hk/~khwong/J2008_IEEE_TIM_Stereo%20Kalman%20.pdf), by Y.K. Yu, et. al.
	- [Vehicle Tracking and Motion Estimation Based on Stereo Vision Sequences](http://hss.ulb.uni-bonn.de/2010/2356/2356.pdf), by A. Barth (long read)
- Deep Learning-based approaches:
	- [Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net](http://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Fast_and_Furious_CVPR_2018_paper.pdf), by W. Luo, et. al.
	- [VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection](https://arxiv.org/abs/1711.06396), by Y. Zhou and O. Tuzel
- Other papers:
	- [Multiple Object Tracking using Kalman Filter and Optical Flow](http://www.ejaet.com/PDF/2-2/EJAET-2-2-34-39.pdf), by S. Shantaiya, et. al.
	- [Kalman Filter Based Multiple Objects Detection-Tracking Algorithm Robust to Occlusion](https://pdfs.semanticscholar.org/f5a2/bf3df3126d2923a617b977ec2b4e1c829a08.pdf), by J-M Jeong, et. al.
	- [Tracking Multiple Moving Objects Using Unscented Kalman Filtering Techniques](https://arxiv.org/pdf/1802.01235.pdf), by X. Chen, et. al.
	- [LIDAR-based 3D Object Perception](https://velodynelidar.com/lidar/hdlpressroom/pdf/Articles/LIDAR-based%203D%20Object%20Perception.pdf), by M. Himmelsbach, et. al
	- [Fast multiple objects detection and tracking fusing color camera and 3D LIDAR for intelligent vehicles](https://www.researchgate.net/publication/309503024_Fast_multiple_objects_detection_and_tracking_fusing_color_camera_and_3D_LIDAR_for_intelligent_vehicles), by S. Hwang, et. al.
	- [3D-LIDAR Multi Object Tracking for Autonomous Driving](https://repository.tudelft.nl/islandora/object/uuid%3Af536b829-42ae-41d5-968d-13bbaa4ec736), by A.S. Rachman (long read)


