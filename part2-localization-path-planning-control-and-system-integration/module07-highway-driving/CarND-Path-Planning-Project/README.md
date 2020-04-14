# **Highway Path Planning for Self-Driving Cars | Pm07**
### Project from the seventh module of the Self-Driving Car Engineer Udacity's Nanodegree

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The aim of this project is to drive autonomously in a highway-like environment as close to the limit speed (50 MPH) as possible, prioritizing safety passing slower cars and slowing down if necessary.

This project uses the [Term 3 CarND Simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2) in order to provide data to the *main.cpp* and to visualize the car behavior, points of the generated trajectory and data from the vehicle.


In particular, the **goals** of this project are implementing the following points in the *src/main.cpp* file:
- **Use the Sensor Fusion Module** data to get useful higher-level information, such as in which lane the other cars are, at which distance they are from us or what is the magnitude of their speed.
- **Implement a Behavior Planning Module** that will use the information about the other cars to decide if the ego car should slow down, change lane to the right or the left, accelerate, etc.
- **Implement a Path Planning Module** that will generate smooth safe, drivable and comfortable trajectories that will follow the lanes, respect the 50 MPH speed limit and try to drive as fast as possible.


The outcome of this project was a program capable of driving a car in a simulated highway-like environment with other cars driving at different speeds following the imposed constraints (detailed in the *Technical Details* section). This can be visualized in the following **YouTube demo**, where the green dots are the 50 next car positions and on the right, some information is shown, such as velocity, accelerations and acceleration derivative (jerk).

[![Demo video](https://img.youtube.com/vi/VXIrqPi8SK4/0.jpg)](https://www.youtube.com/watch?v=VXIrqPi8SK4)


This work will be followed by a brief documentation/overview contained in this file. This project is a completed version of the sample project template provided by the Self-Driving Car Engineer Udemy's Nanodegree. The un-completed original version is [this repository](https://github.com/udacity/CarND-Path-Planning-Project).


## Installation and usage
This repository includes a file that can be used to set up and install [uWebSocketIO](https://github.com/uWebSockets/uWebSockets) for Linux systems (*install-ubuntu.sh*). For Windows you can use either Docker, VMware, or even [Windows 10 Bash on Ubuntu](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) to install uWebSocketIO. Please see the uWebSocketIO Starter Guide page in the classroom within the EKF Project lesson for the required version and installation scripts.

Once the install for uWebSocketIO is complete, the main program can be built and run by doing the following from the project's root directory:

```
mkdir build     # Create a directory for compilation
cd build        # Move to the created directory
cmake ..        # Prepare compilation
make            # Compile
./path_planning # Execute
```


## Dependencies

* cmake >= 3.5
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools]((https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)
* [uWebSockets](https://github.com/uWebSockets/uWebSockets)
  * Run either `install-mac.sh` or `install-ubuntu.sh`.
  * If you install from source, checkout to commit `e94b6e1`, i.e.
    ```
    git clone https://github.com/uWebSockets/uWebSockets 
    cd uWebSockets
    git checkout e94b6e1
    ```


## Technical Details
### Details about my implementation
The car tries to drive at 49.5 MPH in the center lane. When a car is detected under 30 m in front of the ego car, the system tries to pass it using one of the side lanes (if no car is at +-30 m from us on the new target lane). If it is not possible, the car will slow down as much as it is necessary. Humans feel comfortable with accelerations of less than 1G, so the slowing down and speeding ups are performed with accelerations of 5 m/s^2.

The trajectories are estimated using splines generated combining the last points from the last trajectory (to ensure smooth transitions between planning steps) and sparse points far from the ego car. Then, the spline is sampled with 50 points distributed as necessary to respect the limit speed in each 0.02 s time step.

As additional details that deserve consideration is that the lanes are 4 m wide and the track is over 6946 meters. Apart from that, this [spline interpolation library](https://kluge.in-chemnitz.de/opensource/spline/) was used for the trajectory generation, which packs tools for cubic spline interpolation in one single header file (in *./src/spline.h* in this repository). Finally, in the *additional_notes.md* file, there are some more notes I took regarding the project explanation and some suggestions the instructors gave. 

### Details about the simulator
The [simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2) made it possible to abstract the development of this project from the rest of the modules (most of them, already developed in the other projects of the course). It provides *main.cpp* with:

- Ego car's localization data
  - **X** - The car's x position in map coordinates.
  - **Y** - The car's y position in map coordinates.
  - **S** - The car's s position in Frenet coordinates.
  - **D** - The car's d position in Frenet coordinates.
  - **Yaw** - The car's yaw angle in the map.
  - **Speed** - The car's speed in MPH.
- Previous path with the processed points removed:
  - **previous_path_x** - The previous list of x points previously given to the simulator.
  - **previous_path_y** - The previous list of y points previously given to the simulator.
- Previous path's end S and D values
  - **end_path_s** - The previous list's last point's Frenet s value.
  - **end_path_d** - The previous list's last point's Frenet d value.
- Sensor fusion data
  - **sensor_fusion** - A 2D vector of other car's attributes. These attributes are:
    - Car's unique ID.
    - Car's X position in map coordinates.
    - Car's Y position in map coordinates.
    - Car's X velocity in m/s.
    - Car's Y velocity in m/s.
    - Car's S position in Frenet coordinates.
    - Car's D position in Frenet coordinates. 
