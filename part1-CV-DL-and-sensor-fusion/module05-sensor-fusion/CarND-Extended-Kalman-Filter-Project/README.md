# **EKF-based sensor fusion for object tracking**
### Project from the fifth module of the Self-Driving Car Engineer Udacity's Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to estimate the state of a moving object (its X and Y position and velocity) fusing data provided by noisy RADAR and LIDAR measurements using an (Extended) Kalman Filter (EKF) based approach. The target was tu obtain a lower Root Mean Square Error (RMSE) than 0.11 for the target's position (X and Y) and than 0.52 for its velocity (X and Y).

This project uses the [Term 2 CarND Simulator](https://github.com/udacity/self-driving-car-sim/releases) in order to provide data to the *main.cpp* and to visualize this data, the estimated object positions and the Root Mean Square Errors (RMSEs) of the predictions being compared with the ground-truth. 


In particular, the **goals** of this project are the following:
- Implement the *predict* and *update* **functions of the Kalman Filter** (having an *update* version for linear and non-linear measurements).
- Implement the **data fusion** algorithm.
- Implement **helper tools** to compute a jacobian matrix and the RMSE.


The outcome of this project was a data fusion algorithm for its usage with RADAR and LIDAR sensors based in the Extended Kalman Filter. With the provided testing data, it managed to reach RMSEs of 0.0973, 0.0855, 0.4513 and 0.4399 for the target's X and Y positions and velocities (px, py, vx, vy). This can be visualized in the following **YouTube demo**, where the LIDAR measurements are the red circles, the RADAR ones are the blue ones (with an arrow pointing in the direction of the observed angle), and the pose estimations are the green triangles. The simulation is tracking the blue car using RADAR and LIDAR sensors located on the origin of the coordinates system where the car starts its trajectory:

[![Demo video](https://img.youtube.com/vi/Tm7d8E3J3WI/0.jpg)](https://www.youtube.com/watch?v=Tm7d8E3J3WI)


This work will be followed by a brief documentation/overview contained in this file. This project is a completed version of the sample project template provided by the Self-Driving Car Engineer Udemy's nanodegree. The un-completed original version is [this repository](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project).




## Installation and usage
This repository includes a file that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for either Linux or Mac systems (*install-linux.sh*). For Windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project's root directory:

```
mkdir build  # Make a build directory
cd build     # Move inside of it
cmake ..     # Generate compiling files
make         # Compile
./ExtendedKF # Execute
```


## Important Dependencies
* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)


## Testing data
The data used in the simulation (*Dataset 1*) is provided in a text file in the *data* folder of this repository. It consists in a series of a simulated noisy measurements of a LIDAR and a RADAR working asynchronously. Both sensors provide a timestamp in each measurement. The **LIDAR** data are an X and Y positions that represent the post-processed output of a point-cloud that will be the cartesian coordinates of the target object (it obviates the perception step). Regarding the **RADAR**, it gives the polar coordinates of the target (*rho, phi*) and its radial velocity (*rho_dot*). After the data and the timestamp, every measure has listed the ground-truth X and Y positions and velocities of the target.

NOTE: The RADAR-provided velocity is NOT the target's total one, it assumes that the Doppler Effect has been used to compute the radial velocity of the target relative to the RADAR sensor. The tangential relative velocity can not be computed using this procedure, however, I could use this velocity to initialize the target's state so that the initial assumed target's velocity will be a better guess than just a random value.

### Data file format
The information explained above is listed following the next structure:

```
L(for laser) meas_px meas_py timestamp gt_px gt_py gt_vx gt_vy
R(for radar) meas_rho meas_phi meas_rho_dot timestamp gt_px gt_py gt_vx gt_vy
```

Thus, a pair of example data lines would be:

```
R   8.60363 0.0290616   -2.99903    1477010443399637    8.6 0.25    -3.00029    0   
L   8.45    0.25    1477010443349642    8.45    0.25    -3.00027    0
```


## Obtained results
The result after running this EKF and sensor fusion based tracker over the given testing data is the following:

![Visualization of the result]()./readme_images/result.png)

It can be appreciated that, as mentioned before, it managed to reach RMSEs of 0.0973, 0.0855, 0.4513 and 0.4399 for the target's X and Y positions an velocities (px, py, vx, vy). The LIDAR measurements are the red circles, the RADAR ones are the blue ones (with an arrow pointing in the direction of the observed angle), and the pose estimations are the green triangles. The simulation is tracking the blue car using RADAR and LIDAR sensors located on the origin of the coordinates system where the car starts its trajectory.


## Additional resources
Some additional resources can be found in the [GitHub for the  Mercedes Nanodegree](https://github.com/udacity/CarND-Mercedes-SF-Utilities). These are:
- **Matlab scripts** that will generate more data to test the algorithm.
- **Visualization package**.
