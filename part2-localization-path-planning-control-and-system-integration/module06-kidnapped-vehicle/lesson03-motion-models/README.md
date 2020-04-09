# Motion Models and Odometry

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Motion models are the description of the physics of a vehicle. A mathematical model describing how the car will behave given a certain motion command (giving gas, steering the wheel, etc.). They are useful to predict where the car will be in the near future, which can be used to track or localize it. We will study the ***bicycle model***.

## Bicycle Model
It is a simple way of modeling a car movement. It makes several assumptions:

- Ignores all vertical car dynamics (the car will only move in 2D).
- The front wheels are connected to the back ones by a rigid beam with fixed length.
- We represent the 2 front wheels as only one that will be on the center of the car and the same for the 2 back ones (they act together by pairs).
- The control will be a turn angle *theta* of the front wheel and a longitudinal velocity in the direction the car is facing.

As a refresher, the *yaw* angle (*theta*) of a car is its direction observing from the top. Assuming that it is constant, the motion model equations will be:

```
# Final position using the initial one and the velocity during the elapsed time (dt)
x_f = x_0 + v(dt) * cos(theta_0)
y_f = y_0 + v(dt) * sin(theta_0)

# Assumption: Constant turn angle (during the elapsed time)
theta_f = theta_0
```

NOTE: If we are driving on very hilly roads, the car's *pitch* can also be very relevant for the motion model.


## Odometry and its limitations
It is another way of know how much the car or some other kind of mobile robot has moved. This is usually done measuring how many times a wheel turns, commonly by encorders. Knowing this and the wheel radius, we can compute the advanced distance. This can be done by:

```
# Final position using the initial one, the number of turns and the wheel circunference
x_f = x_0 + n_turns * cos(theta) * wheel_circ
y_f = y_0 + n_turns * sin(theta) * wheel_circ
```

This will be unaccurate many occasions, such as if the wheel slides (due to, for example, wet paviment) or when there are a lot of bumps in the road, because the wheels will turn spending some of their circunference going up and down the bump, not only advancing in the direction the car is facing.

