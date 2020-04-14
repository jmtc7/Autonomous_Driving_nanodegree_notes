# **PID Controller for Lane Positioning for Self-Driving Cars | Pm08**
### Project from the eight module of the Self-Driving Car Engineer Udacity's Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to implement a PID controller that will determine the required steering angle in order to keep a car with constant speed in the center of a single-lane track while driving at a 30 MPH speed.

This project uses the [Term 2 CarND Simulator](https://github.com/udacity/self-driving-car-sim/releases) in order to provide data to the *main.cpp* and to visualize the car behavior and data from the vehicle.


In particular, the **goals** of this project are implementing the following points in the *src/main.cpp* and *src/PID.cpp* files:

- PID **controller** in the *PID.cpp* file, consisting in:
  - Initialization
  - Error computation
  - Error combination
- PID **usage** in the *src/main.cpp* file.
- PID gains **fine tunning**.
  - Explained in the *Gain Tunning* section of this file.


The outcome of this project was a PID controller capable of correct the steering angle of an autonomous car driving at 30 MPH. This can be visualized in the following **YouTube demo**:

[![Demo video](https://img.youtube.com/vi/FqoRhr8x-64/0.jpg)](https://www.youtube.com/watch?v=FqoRhr8x-64)


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


## Gain Tuning
Even the ***Twiddle*** algorithm was proposed on the course as a good Coordinate Ascent (it is explained in this [video](https://www.youtube.com/watch?v=2uQ2BSzDvXs) by Sebastian Thrun), a different strategy was used for the parameter tuning of this project. 

This strategy consisted in the following steps (to get the first set of gains to fine-tune):

- Start will all the control gains at zero.
- Increase the *proportional* gain `Kp` until the car is able to drive more or less properly with constant oscillations. The fine tuning of this was done with increments of 0.0025. With `Kp` over 0.0100, the car was kind of able to drive, even it was struggling in the curves and showed an obvious **systematic bias** towards the right that made it drive over the right lane line.
- Increase the *differential* gain `Kd` to reduce the oscillation to make it more robust when driving close to the lane limits. When it reached 0.0075, it was possible to raise `Kp` until 0.0150 and still have a decent response. However, due to the systematic bias, the car was driving very close to the edges, so it missed the road from time to time.
- Finally, increase the *integral* gain `Ki` to fine-tune the controller in order to try to make the response quicker (even if it caused some overshooting) and to correct the already mentioned offset. Even the smallest increase of this gain had a big effect in oscillations, so `Kd` was increased up to 0.5000 to compensate so. `Kp` was reduced back to 0.0010 so that it will contribute less to generate these oscillations.

After tweaking them a bit more, `Kp=0.0100`, `Ki=0.0001`, and `Kp=0.6000` were the parameters that I ended up with. From this point in advance, I just tweaked them to try to get better performance. What I kept in mind during this more empiric phase of the tunning was:

- Increase `Kp` adds velocity and oscillation.
- Increase `Ki` adds more velocity and oscillation than `Kp` and also makes the car to drive more centered (removes the systematic bias).
- Increase `Kd` reduces the oscillations and the velocity.

I saw that the car was not oscillating much, but rather the response was so slow that these small oscillations made it go off the road. This is why I decided to boost all the gains up to `Kp=0.0150`, `Ki=0.0020`, and `Kp=1.5000`. These values, after some further fine tunning, became the next definitive ones: `Kp=0.0150`, `Ki=0.0025`, and `Kp=2.7500`.

Obviously, the system's response will not be the same for different velocities. I tried to adjust it for a velocity of 30 MPH, even it can obviously be used for other velocities. However, specially for higher ones, it may have a more inappropriate behavior, such as more abrupt oscillations.

