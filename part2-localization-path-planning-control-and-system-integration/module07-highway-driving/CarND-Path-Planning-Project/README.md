# My Additional Notes
## Getting Started
In the project, a car will be self-driving in a highway at a maximum speed of 50 MPH. The car will provide us with the outputs of its Localization and Data Fusion Modules. This last one will give the location of all vehicles on the side of the road where the ego car is.

The planner's output is a list of X and Y global map coordinates. The car will advance from one point to the other each 20ms and its orientation will be the result of connecting the current point with the last one.

The starting point was to make the car drive in a straight line at the given speed limit. To do this, in *main.cpp*, the *next_x_vals* and *next_y_vales* are forwarded to the simulator (instead of the velocity). In order to get a 50 MPH velocity, the advance should be done at over 25 m/s, which means that it is necessary to forward advances of 0.5 m each timestep (will be achieved in 20ms).

This, however, implies that the car will go from 0 to 65 MPH in a single timestep, which leads to very high acceleration and jerk. In order to avoid this, we want to limt the acceleration to 10m/s^2 (~1G) and the jerk to 10m/s^3.

The next improvement was use some point from the previous path to ensure a smooth trajectory. It is important to consider how many of them are going to be used because it is a must to reflect the changes in the environment in our trajectory and we may need new points to do so.

A relevant consideration to do is that the C++ program will get data from the simulator in one cycle and compute and send the trajectory in the next one. This means that, by the time the trajectory reaches the simulator, the car will be likely to have gone through a few points of it. However, the simulator was designed with this in mind, so the car will go towards the first relevant point, but it is something to keep in mind if this is implemented in other circumstances.

The *getXY()* function converts from Frenet Coordinates (*s, d*) to the traditional Cartesian ones (*x, y*).

The *sensor_fusion* variable contains all the information from the cars that are traveling in our direction. For for each car, the information will be: *[id, x, y, vx, vy, s, d]*. The id is a unique identifier for that car. The x, y values are in global map coordinates, and the vx, vy values are the velocity components, also in reference to the global map. Finally s and d are the Frenet coordinates for that car.

Regarding **lane changes**, they should be considered whenever the ego car is following another car that is driving too slowly. This change, however, will only happen if it is safe and makes the ego car be able to move better through the traffic flow. Moreover, the system should maximize the distances to other cars for safety and minimize acceleration and jerk for comfort. The lane 0 will be the one on the left and we will be going to other ones until the 5th lane. We will be driving from the 3rd to the 5th.

In this [video](https://www.youtube.com/watch?v=7sI3VHFPP0w), the original developer of the code goes through it explaining it. Some relevant time marks:

- 00:03:20 - distance(), ClosestWaypoint()
- 00:03:55 - getFrenet()
- 00:05:20 - getXY()
- 00:09:00 - Drive in straight line with constant velocity
- 00:11:30 - Stay in our lane using Frenet Coordinates (full code at 16:35). Decrease speed lowering the distance we do in each timestep.
- 00:19:00 - **Smooth the path** using jerk minimization and trajectory generation. We can also use *splines* to make sure we go through all the points. Splines may be more convinient because of this [spline library](https://kluge.in-chemnitz.de/opensource/spline/).
- 00:21:00 - Include the **splines library**. We could:
  - Create a spline using our XY points extracted from the Frenet-cartesian coordinate conversion.
  - Create a spline using points far from each other and sample it to create our trajectory. This is what they do. File in 22:25 (*aaron.cpp*), code in 23:30.
    - Chose a lane.
    - Set target velocity.
    - Use the two last points from the last trajectory (if it has less than 2 points, the last car state will be used). 
      - These are the points that, for whichever reason, the car did not went through in the last trajectory.
    - Get state of the car (x,y,theta) in the used points from the last trajectory.
    - 00:27:55 - Generate points at 30, 60 and 90 meters away from the current car state.
      - NOTE: Now we have the two last points from the last trajectory and the new 3 widely spaced ones.
    - 00:29:30 - spline creation
      - It will be sampled to get 50 points in each trajectory generation
    - 00:33:00 - How to determine how many points we need to sample from the spline given our target final point and velocity during the trajectory.
      - implementation at 00:35:45
      - NOTE: Target velocity divided by 2.24 in order for it to be in m/s instead of MPH.
    - 00:38:15 - move from local to global coordinates
- 00:40:00 - Car is driving smoothly, but need to **solve** (1) cold start (from 0 to 50 MPH in 0.2s) and (2) **collisions**.
  - 00:41:05 - Use the sensor fusion list to avoid hitting cars in front of the ego car (i.e. evaluate if a car is in our lane and, if it is too close, do something).
  - Each element in *sensor_fusion* variable is a car on the road in our driving direction. We use *d* (Frenet coordinates) to know which of them are in the same lane as the ego car.
  - 00:46:20 - If the ego car is too close to the another car on its lane (<30 m), we will do something (e.g. lowering the speed to 30 mph).
- 00:48:15 - New problem: Once a car is detected, the system is never told to **increase** its **velocity after** the **reduction**. Moreover, the ego car suffers a lateral collision.
- 00:48:45 - **Solve the cold start** (ego car raises its velocity from 0 to 50 MPH in 0.2s).
  - This is solved, alongside not rising the velocity after detecting a car too close, by using a certain flag that will tell the system when it needs to slow down (i.e. there is something slower than the ego car in front of it) and when it should accelerate (i.e. the obstacle is far enough, we need to rise speed, etc.). We do this by adding increments of speed in a way that the system will not require accelerations or jerks higher than 10 m/s^2 or 10 m/s^3 (0.224 m/s for points each 0.2 seconds).
  - 00:49:50 - To fix the cold start with this, the initial reference velocity should be 0.0 mph so it will be raised slowly.
  - 00:50:50 - This changes our velocity for the whole cycle. We could edit it in a way that the reduction/increase is performed for each point of the trajectory, which will allow a faster driving.
- 00:52:40 - New problem: Try to **pass slow cars** instead of just slowing down.
  - Where we detected if another car was too close to us (using the sensor fusion data), we can not only set the flag to slow down our car, but also change the target lane, so in the spline creation, the target will be this new specified lane.
  - Moreover, it is also essential to check (with the sensor fusion data) if we have enough gap in our target lane to change to it in order to avoid collisions.
    - Finite State Machines and/or cost functions may help making this decision making more advanced.
    - Maybe Gausian Naive Bayes classifiers could help the system to decide which is the best lane to be in 5-10 seconds in the future.
  - It is necessary to keep in mind that we do not want to drive in the opposite direction or off of the road.
- 00:58:45 - **Answering students questions** 
  - 01:00:00 - This walkthrough was done in order to link better the lessons of the module and the final project. They give some additional thoughts about how to approach the project.
  - 01:04:25 - **Spline** implementation review:
    - Initialize it (*tk.spline s;*).
    - Forward the anchor points or *knots* (*s.set_points(ptsx, ptsy)*).
    - Use generated spline to get points (*y = s(x)*).
    - NOTE: If the spline gets too vertical and for one X value we get several Y ones, we will run into problems. That is why the *double polymerization transformation* was done, which will turn the spline horizontal so we will have several equal X values instead of equal Ys.
  - 01:06:40 - Strategies to **smooth trajectories between calculations**:
    - Using the previous path points should eliminate any problems with high jerks.
    - Use a quintic polynomial generation instead of a spline
  - 01:09:00 - A mentor suggested to transform the waypoints from global coordinates to local ones.
    - Without a map to localize the car in, it is no doable.
    - Maybe he was talking about shifting the car orientation to 0 degrees, as is done in the line 367.
      - We need to undo this as in line 414!
  - 01:10:35 - How does finite state machines work with hybrid a-star?
    - Hybrid a-star algorithm does not makes much sense for such an structured environment, it would be more useful in something like parking. 
    - For a highway, the system needs continuous decisions, so a finite state machine or cost functions are better approaches for this applications.
  - 01:12:00 - Why is *s* included in the map landmarks list? Is not it very easy to compute?
    - It could be computed or obtained just by using one of the helper functions, but this way it is just easier.
  - 01:12:40 - Could we use Frenet coordinates for both trajectory generation and behavior planning?
    - Yes. For trajectory generation it will be necessary to go back to XY coordinates to be able to control the car, though.
  - 01:13:30 - How to deal with the inaccurate acceleration from the simulator?
    - Measuring it from velocity differences between timesteps could be a good options.
    - However, with quintic polynomial generations, it is common to use 0 starting and ending accelerations.
  - 01:14:45 - How to compute costs for different possible next states? Such as for comparing which is the most convenient lane to be in in X seconds.
    - It is possible to generate different splines using different anchor points and calculate the jerk for those splines and select the one with the minimum jerk.
  - 01:16:35 - What logic to include in a *prepare change lane x* state?
    - These states are useful for when the ego car wants to change lane but there is no gap on it.
    - Our actions should depend on what the car that is not allowing us to move is doing.
      - If it is accelerating, the ego car may slow down, while if it is slowing down, the system should accelerate.
  - 01:18:15 - What does the *angle* variable in function *nextWaypoint()* represents?
    - It is the car's angle. i.e. It is useful to know which waypoint it is looking at. The car may be closer to a waypoint that is behind it than to the actual next one.
    - The comparison with *pi/4* is because the waypoint needs to be in this 45 degrees view angle.
  - 01:19:00 - What are the key concepts of the project? e.g. Jerk-Minimizing Trajectories (JMT), cost functions, Finite State Machines (FSMs). How to avoid violations of jerk and speed limits?
    - The minimal used things to use are: Frenet coordinates, a path smoother (e.g. quintic polynomial or splines) and a FSM to figure out which maneuver to perform.
  - 01:19:50 - How to combine JMT and splines to get a solution?
    - We need to choose one of them. Both methods are used for smoothing the paths.
  - 01:20:50 - How to calculate *s_dot_dot*?
    - It is the second derivative os *s*, which is the ego car's position in Frenet Coordinates.
    - We compute *s_dot* by taking the distance between two consecutive *s* and divide it by the elapsed time (probably 0.2s).
    - To compute *s_dot_dot*, it is possible to do the same using the difference between two *s_dot* and the elapsed time.


## Waypoints in the Map
The *data/highway_map.csv* waypoint list will be used. It contains 181 waypoints distributes across a track that is 6945.554 meters long (over 4.32 miles). Thus, assuming a velocity of over 50MPH, it should be completed in a bit more of 5 minutes. The track has 6 lanes, 3 in each direction, each of them 4 meters wide. 

Each waypoint has an (x,y) global map position, and a Frenet s value and Frenet d unit normal vector (split up into the x component, and the y component). 

The d vector has a magnitude of 1 and points perpendicular to the road in the direction of the right-hand side of the road. The d vector can be used to calculate lane positions. For example, if you want to be in the left lane at some waypoint just add the waypoint's (x,y) coordinates with the d vector multiplied by 2. Since the lane is 4 m wide, the middle of the left lane (the lane closest to the double-yellow dividing line) is 2 m from the waypoint.

If you would like to be in the middle lane, add the waypoint's coordinates to the d vector multiplied by 6 = (2+4), since the center of the middle lane is 4 m from the center of the left lane, which is itself 2 m from the double-yellow dividing line and the waypoints.

In order to generate trajectories between the waypoints, it will be necessary to interpolate to get trajectory points that are between each pair of consecutive waypoints. In the lessons of this module, polynomial fitting was studied (Polynomial Trajectory Generators or PTGs), but there are other approaches that can solve this problem, such as *Bezier curve fitting with control points*, or *spline fitting*, which guarantees that the generated function passes through every point. [This](https://kluge.in-chemnitz.de/opensource/spline/) is a useful C++ tool for splines.


---
---
---




# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program
   
### Simulator.
You can download the Term3 Simulator which contains the Path Planning Project from the [releases tab (https://github.com/udacity/self-driving-car-sim/releases/tag/T3_v1.2).  

To run the simulator on Mac/Linux, first make the binary file executable with the following command:
```shell
sudo chmod u+x {simulator_file_name}
```

### Goals
In this project your goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

#### The map of the highway is in data/highway_map.txt
Each waypoint in the list contains  [x,y,s,dx,dy] values. x and y are the waypoint's map coordinate position, the s value is the distance along the road to get to that waypoint in meters, the dx and dy values define the unit normal vector pointing outward of the highway loop.

The highway's waypoints loop around so the frenet s value, distance along the road, goes from 0 to 6945.554.

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./path_planning`.

Here is the data provided from the Simulator to the C++ Program

#### Main car's localization Data (No Noise)

["x"] The car's x position in map coordinates

["y"] The car's y position in map coordinates

["s"] The car's s position in frenet coordinates

["d"] The car's d position in frenet coordinates

["yaw"] The car's yaw angle in the map

["speed"] The car's speed in MPH

#### Previous path data given to the Planner

//Note: Return the previous list but with processed points removed, can be a nice tool to show how far along
the path has processed since last time. 

["previous_path_x"] The previous list of x points previously given to the simulator

["previous_path_y"] The previous list of y points previously given to the simulator

#### Previous path's end s and d values 

["end_path_s"] The previous list's last point's frenet s value

["end_path_d"] The previous list's last point's frenet d value

#### Sensor Fusion Data, a list of all other car's attributes on the same side of the road. (No Noise)

["sensor_fusion"] A 2d vector of cars and then that car's [car's unique ID, car's x position in map coordinates, car's y position in map coordinates, car's x velocity in m/s, car's y velocity in m/s, car's s position in frenet coordinates, car's d position in frenet coordinates. 

## Details

1. The car uses a perfect controller and will visit every (x,y) point it recieves in the list every .02 seconds. The units for the (x,y) points are in meters and the spacing of the points determines the speed of the car. The vector going from a point to the next point in the list dictates the angle of the car. Acceleration both in the tangential and normal directions is measured along with the jerk, the rate of change of total Acceleration. The (x,y) point paths that the planner recieves should not have a total acceleration that goes over 10 m/s^2, also the jerk should not go over 50 m/s^3. (NOTE: As this is BETA, these requirements might change. Also currently jerk is over a .02 second interval, it would probably be better to average total acceleration over 1 second and measure jerk from that.

2. There will be some latency between the simulator running and the path planner returning a path, with optimized code usually its not very long maybe just 1-3 time steps. During this delay the simulator will continue using points that it was last given, because of this its a good idea to store the last points you have used so you can have a smooth transition. previous_path_x, and previous_path_y can be helpful for this transition since they show the last points given to the simulator controller with the processed points already removed. You would either return a path that extends this previous path or make sure to create a new path that has a smooth transition with this last path.

## Tips

A really helpful resource for doing this project and creating smooth trajectories was using http://kluge.in-chemnitz.de/opensource/spline/, the spline function is in a single hearder file is really easy to use.

---

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

## Editor Settings

We've purposefully kept editor configuration files out of this repo in order to
keep it as simple and environment agnostic as possible. However, we recommend
using the following settings:

* indent using spaces
* set tab width to 2 spaces (keeps the matrices in source code aligned)

## Code Style

Please (do your best to) stick to [Google's C++ style guide](https://google.github.io/styleguide/cppguide.html).

## Project Instructions and Rubric

Note: regardless of the changes you make, your project must be buildable using
cmake and make!


## Call for IDE Profiles Pull Requests

Help your fellow students!

We decided to create Makefiles with cmake to keep this project as platform
agnostic as possible. Similarly, we omitted IDE profiles in order to ensure
that students don't feel pressured to use one IDE or another.

However! I'd love to help people get up and running with their IDEs of choice.
If you've created a profile for an IDE that you think other students would
appreciate, we'd love to have you add the requisite profile files and
instructions to ide_profiles/. For example if you wanted to add a VS Code
profile, you'd add:

* /ide_profiles/vscode/.vscode
* /ide_profiles/vscode/README.md

The README should explain what the profile does, how to take advantage of it,
and how to install it.

Frankly, I've never been involved in a project with multiple IDE profiles
before. I believe the best way to handle this would be to keep them out of the
repo root to avoid clutter. My expectation is that most profiles will include
instructions to copy files to a new location to get picked up by the IDE, but
that's just a guess.

One last note here: regardless of the IDE used, every submitted project must
still be compilable with cmake and make./

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

