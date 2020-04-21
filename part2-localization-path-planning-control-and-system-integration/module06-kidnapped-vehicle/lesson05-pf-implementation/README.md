# C++ Implementation of Particle Filters

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Particle Filter Steps
### Initialization
We will most likely use the GPS localisation and introduce the sensor noise. We will also define some critical hyperparameters of the filter, such as the number of particles to use. Too few particles will make it harder to localize precisely the car, while too many will increase the computational requirements.

We will need to put particles covering the places where our car could be. Without any other knowledge, we should sample the whole world, but we can use GPS to estimate a way smaller zone to be sampled. We can add a normal distribution with the mean in the GPS-provided location and a covariance depending on its accuracy or the trust we have on its measurements. To do so, we can use the [C++ standard library normal distribution](https://en.cppreference.com/w/cpp/numeric/random/normal_distribution) and the [C++ standard library random engine](http://www.cplusplus.com/reference/random/default_random_engine/).

This step is implemented in the *00_sampling.cpp* file, using 2 m of uncertainty in X and Y for the GPS location.

### Prediction Step
We apply the motion model (using the yaw rate and velocity) to each particle. We will add gaussian noise to both the velocity and yaw rate because the robot's movement may be innacurate. We implmement this by simply using the motion model. We should use the general equations if the yaw rate is not 0.

### Update Step
This consists in computing the importance weights of the particles. We use the map to know where the landmarks should be for each particle and the observation made by the robot to know how far/close each particle is from the expected observation.

#### Data Association
Before using the detected features from the robot observation, we need to match them with the feature detection associated to each particle (i.e. the *pseudo-ranges*). To do this matching, we can use the simples possible approach, *Nearest Neighbour*, which consist in choosing as a correspondance the nearest measurement to the landmark.

This matching technique works better when we have sensors with low noise and an accurate motion model. This way, the measurements will correspond to objects very near to the detection and we will be able to locate them preciselly because the car will be close to where we think it is (the measurement will be relative to the car position).

However, if we use sensors that provides us with very dense data, we might miss-associate some data. Another downside of this method is its efficiency. For each landmark, we will have to measure the distance to each measurement. Obviously, the noisy data will introduce a high chance of miss-matching data. Moreover, if the vehicle position estimation is too bad, given than the measurements are relative to it, we will most likely miss-match them with the map's features. Finally, it does not consider the uncertainty of each sensor. Maybe a sensor is very precise on the tangential direction to the measurement and have more uncertainty in the radial one. In this case, we should trust more an association if the map feature is in the line connecting the car and the measurement, even if other measurement is closer but more to the right or left.

#### Weight Computation
Once having the correspondences of the data, we can use the *Multivariate Gaussian Probability Density* function for each measurement and combine the likelihoods of each one of them to compute the importance weight of each particle. It assumes gaussian noise in the measurements and independence between them.

### Resampling
Finally, we will resample our particle collection according to its importance weights. After the resampling, we will go back to the prediction step.


## Evaluation measures
Once having the Particle Filter implemented, we may want to measure its accuracy using a ground-truth to quantify how good our fine-tuning is or to evaluate how much we can trust it. There are several ways of doing it:

- Mean Weighted Root Square Error: Compute the RSE between each particle and the ground-truth and multiply it by the particle weight. Finally, compute the mean of all these weighted errors dividing by the sum of the weights. Remember that the error will take into account X, Y and *theta*.
- Root Square Error of the best particle.


## Implementation Details
In the following quizes, we will be using one only particle in order to work with the coordinate system tranformation (global-car) and work positioning the measurements in the global map coordinates. For doing so, we will:

- Transform the car's landmark observation from the car coordinate system to the map one.
- Associate the transformed observations with the map's landmarks.
- Update the particle's weights by:
  - Applying the Multivariate Gaussian Probability Density function (implemented in the *02_multivariate_gaussian_prob* folder).
  - Combining the probabilies of each observation multiplying them to get the total probability of the observation of the particle that is being processed.

### Car-to-Map Transformation
This will be done by a matrix multiplication known as *homogeneous transformation*, which consists on a rotation followed by a translation. In this transformation matrix, the two first columns and rows correspond to the rotation and the two first rows of the last column, to the translation (assuming a 2D space).

Some additional resources related to this are:

- [Coordinate Transformations](http://farside.ph.utexas.edu/teaching/336k/Newtonhtml/node153.html)
- [Coordinate Transformation Under Rotation](https://www.miniphysics.com/coordinate-transformation-under-rotation.html)


## Further Reading
- Simultaneous Localization and Mapping (SLAM):
  - [Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/abs/1606.05830) by C. Cadena, et. al.
  - [Navigating the Landscape for Real-time Localisation and Mapping for Robotics and Virtual and Augmented Reality](https://arxiv.org/abs/1808.06352) by S. Saeedi, et. al.
- Other methods:
  - [Robotic Mapping: A Survey](http://robots.stanford.edu/papers/thrun.mapping-tr.pdf) by S. Thrun. From 2002.
