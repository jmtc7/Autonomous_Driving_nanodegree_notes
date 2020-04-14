# **PID Controller for Lane Positioning for Self-Driving Cars | Pm08**
### Project from the eight module of the Self-Driving Car Engineer Udacity's Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to implement a PID controller that will determine the required steering angle in order to keep a car with constant speed in the center of a single-lane track while driving at a 100 MPH speed.

This project uses the [Term 2 CarND Simulator](https://github.com/udacity/self-driving-car-sim/releases) in order to provide data to the *main.cpp* and to visualize the car behavior and data from the vehicle.


In particular, the **goals** of this project are implementing the following points in the *src/main.cpp* and *PID.cpp* files:

[TODO]


The outcome of this project was a PID controller capable of correct the steering angle of an autonomous car driving at high speed (100 MPH) in a track with a single lane. This can be visualized in the following **YouTube demo**:

[![Demo video](https://img.youtube.com/vi/TODO/0.jpg)](https://www.youtube.com/watch?v=TODO)


This work will be followed by a brief documentation/overview contained in this file. This project is a completed version of the sample project template provided by the Self-Driving Car Engineer Udemy's Nanodegree. The un-completed original version is [this repository](https://github.com/udacity/CarND-Path-Planning-Project).


## Installation and usage
This repository includes a file that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for Linux systems (*install-ubuntu.sh*). For Windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project's root directory:

```
mkdir build # Create a directory for compilation
cd build    # Move to the created directory
cmake ..    # Prepare compilation
make        # Compile
./pid       # Execute
```

Tips for setting up your environment can be found [here](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/23d376c7-0195-4276-bdf0-e02f1f3c665d)


## Dependencies

* cmake >= 3.5
 * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1(mac, linux), 3.81(Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `./install-mac.sh` or `./install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```
    Some function signatures have changed in v0.14.x. See [this PR](https://github.com/udacity/CarND-MPC-Project/pull/3) for more details.
* Simulator. You can download these from the [project intro page](https://github.com/udacity/self-driving-car-sim/releases) in the classroom.

Fellow students have put together a guide to Windows set-up for the project [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/Kidnapped_Vehicle_Windows_Setup.pdf) if the environment you have set up for the Sensor Fusion projects does not work for this project. There's also an experimental patch for windows in this [PR](https://github.com/udacity/CarND-PID-Control-Project/pull/3).


## Technical Details
[TODO?]

