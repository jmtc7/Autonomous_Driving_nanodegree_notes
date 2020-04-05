# Kalman Filters

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In order to track something's position, there are two main approaches: (1) the Kalman Filters and (2) the Monte Carlo Localization (Particle Filters). The main difference between them is that the KFs provide a continuous and multimodal probability distribution of the target's pose while the PFs provide discrete, multimodal probabilities.

The Kalman Filters predict where the target will be given certain past observations. They use Gaussians (distributions centered in a *mean* and with certain *covariance* in which the area underneath sums 1). If we visualize a gaussian, it will be symetrical relative to the mean and unimodal (will only have one peak) and will have an exponential raising and falling.

The bigger the (co)variance is, the less certain we are about the pose of the tracked target.

The Gaussian equation is: `y = f(x) = (1/(sqrt(2*pi*sigma2))) * exp((-1/2)*((x-mu)**2/sigma2))`, being *sigma2* the *covariance* and *mu* the *mean*.

Kalman Filters are divided in two steps: (1) the measurement update and (2) the motion uptdate or prediction. In the **measurement update** we estimate where we are using our last position and the new perceived data. During the **prediction** step, we update our current belief in our position using the motion command that has been executed, *predicting* where we should be.


## Velocity estimation with gaussians
Even if our sensor is only capable of measuring the position of an object, we can use gaussians in a 2D space in which one axis is the position and the other one the velocity. This way, we can represent certainty about the position and uncertainty about velocity with a high variance in one of the gaussian's axis.

If we can build a gaussian able to represent the relation between where I will be if I had a certain velocity. This does not tell much with only one observation because I will have too much velocity uncertainty to know my position and too much position uncertainty to know my velocity. BUT when we add a new position gaussian (with velocity uncertainty very big), we can make the product of this relationship gaussian and the new observation and I will know with a finite covariance both my position and my velocity.

## Kalman filter states (variables)
We wil have observable states (such as position) and hidden ones (e.g. velocity). However, since these states interact between eachother, we can inferr the hidden ones using a sequence of observation of the observable states.

