# PID Control

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Control is how to use the (already obtained) steering and throttle to more a real car where desired. 

## Towards the PID Controller
### Proportional Control (P)
We can use controllers to keep driving in the desired position, assuming that the system is able to know where it is driving and where it wants to be driving. In order to do this, the *Cross Track Error* (CTE) is computed (distance between the current position and the desired one) and a correction proportional to it will be performed in the steering angle (i.e. the bigger the error, the more the car will turn).

The weakness of this simple approach is that the car will overshoot. Whenever it reaches the reference (desired) position, it will still be slightly turned, not aligned with the reference line, so it will not be able to just reach the reference and stay there.

The *00_p_controller.py* scripts imlements a P controller in which a robot starts being at [0,1] oriented towards the X axis and the target is to be driving alongside the X axis. Therefore, the CTE is the Y coordinate. A parameter `tau` is used to correct the trajectory proportionally tho the CTE. When experimenting with the control parameter `tau`, the oscilations will become bigger, but in each step the error will decrease quicker.

### Proportional-Differential Control (PD)
In order to avoid the overshooting of the P controllers, it is possible to use, appart of out proportional gain `tau_p`, a differential gain `tau_d` that will regulate the impact of the temporal derivative of the CTE. As the error decreases, the derivative will be smaller and smaller, which will make the corrections smoother. In code, it is possible to perform a *discrete derivative* using `(CTE_t - CTE_t-1)/delta_t`. i.e. The difference of CTE divided by the elapsed time between the measurements. In the implementation, this elapsed time will be *1 time step*, so the control law will be:

```
aplha = - tau_p*CTE - tau_d*(current_CTE - last_CTE)
```

This is implemented in the *01_pd_controller.py* script. It shows the outputs of the PD controller and its comparison with the P controller one.

Even the overshooting problem is pretty much solved, the ***Systematic Bias*** problem arises. This consist in possible mechanical imperfections, such as the wheels being slightly turned to one of the sides, that will end up introducing a bias or offset in the final response of our system. i.e. instead of being stabilized in Y=0, it will stabilize itself at another point with a certain CTE (depending on the magnitude of the *imperfection*).

### Proportional-Differential-Integral Control (PID)
This leads us to add a new term to the control law, now using the accumulation of the CTEs over time. This is the integral of the error. This way, if the car has a certain constant error, it will be accumulated, making the system to try to correct it with harder corrections each time.

This is implemented in the *02_pid_controller.py* script. It shows the outputs of the PID controller and its comparison with the ones from P and PD controllers. The systematic bias is added ussing the *drift*.


## Twiddle: Coordinate Ascent for Parameter Optimization
When using a PID controller, there are 3 parameters that require tuning to optimize the controller's performance. In order to do this, we can use *Coordinate Ascent*, also known as ***Twiddle Algorithm***.

In order to optimize the parameters, the `run()` function must return a *goodness score* to be able to evaluate its performance. This could be the average CTE if the target is to minimize this error, given that it will depend on the three control gains to tune. Once the `run()` function is adapted, we can start applying Twiddle by:

- Creating a vector to store the three parameters to optimize and initialise it with zeros (*[0, 0, 0]*).
- Creating a vector of potential changes to be done to these parameters. For now, they will be ones (*[1, 1, 1]*).
- Execute the `run()` function with the initial parameters to get an initial *best error*.
- Execute the function with the initial parameters with the potential changes added.
- If the *goodness* is better with the potential changes:
  - Store the new best error.
  - The potential changes will be modified.
    - Make them more distant to the initial ones.
    - Multiply them by 1.1.
- If the result was better with the initial parameters:
  - Substract the potential changes twice (because they were added once before).
- If this (substractin modification from the initial value) is also worst than the initial value:
  - Return parameters to the initial values by adding the potential change once.
  - Reduce the modification by multiplying them by 0.9.
- Iterate until the sum of the modifications is smaller than a threshold (the modifications will be too small).

This algorithm is implemented in the *03_twiddle.py* script. It displays a graph with the output of the twiddle-tunned PID controller and a comparison between the outputs of the P, PD, PID and twiddle PID controllers.


## Additional reading
- Model Predictive Control (MPC): [Vision-Based High Speed Driving with a Deep Dynamic Observer](https://arxiv.org/abs/1812.02071), by P. Drews, et. al.
- Reinforcement Learning-based: [Reinforcement Learning and Deep Learning based Lateral Control for Autonomous Driving](https://arxiv.org/abs/1810.12778), by D. Li, et. al.
- Behavioral Cloning: [ChauffeurNet: Learning to Drive by Imitating the Best and Synthesizing the Worst](https://arxiv.org/abs/1812.03079), by M. Bansal, A. Krizhevsky and A. Ogale.
