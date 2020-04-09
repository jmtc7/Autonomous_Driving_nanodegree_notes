# Particle Filters

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


## Particle Filters VS Histogram and Kalman filters
This is the third algorithm for estimating the state of a system, the most common ones are the Histogram Filter, the Kalman Filter and the Particle Filter. The HF has a discret state space, an multimodal belief and its efficiency is exponential with the number of dimensions (any grid defined over K dimensions will have exponentially many grid cells with K). The KF has a continuous state space, an unimodal belief and its efficiency is quadratic (we use a mean vector and the covariance matrix). Both of them provide approximations. The HF because what they represent is usually not discrete and the KF because it is only exact for linear systems, which is usually not the case.

Regarding Particle Filters, their state space is continuous, their belief multimodal and they provide approximate solutions. Regarding their efficiency, it varies depending the domain of applications. Sometimes they scale exponentially, so we should avoid applying it with more than 4 dimensions. However, in domains such as tracking, they scale much much better.

However, the key advantage of PFs is how easy they are to program. We can assume a robot taking distance measuremets with a SONAR sensor, for example. Then, the PF will be implemented using *particles*, which are hypothesis of where the robot is. When we are completelly uncertain about where it is, the map will be filled with particles more or less equally distributed (every hypothesis will be as good as every other possible one). We will define how many *particles* (hypothesis) we want to keep in each moment. Each particle wil contain the X and Y pose coordinates and the orientation *theta*. Whenever we take a new measurement, the particles whose observations are closer to the ones received by the actual robot will have a higher chance to survive, while the ones whose pseudo-ranges are farer away will be more likely to disappear. After each step, we will remove some particles and *resample* the map, usually creating more particles in states (X, Y, *theta*) close to the ones of the particles with higher marks.


## Particle Filter Implementation
We will be using the class *Robot* provided in the course. In the *00_sample_robot_class.py*, it is implemented alongisde some sample code in which we add noise to the measurements, define a robot and its initial position and sends some movement commands and makes a couple of observations.

To generate a set of particles, we will define how many we want and we will create a list with this *N* number of instances of the *Robot* class. We can then apply motions or get measurements of every particle with a simple for loop. 

The second (and hardest) part of the PF implementation consists in assign ***importance weights*** to the particles. We will use these weights to decide which one survives and which one is destroyed. These weights are computed by comparing the result of the real robot observation and the result of the observation returned by each particle. The closer these observations are to each other, the highest importance weight the particle will have and the more likely will be to keep it.

Once the importance weights have being computed and assigned, we will keep each particle with a probability proportional to its importance weight. After this, it is time for the final PF step: ***resampling***. In order to do this, we will start by normalizing all the importance weights (dividing them by the sum of all of them). We will then re-build a new set of particles choosing particles from the old set. Obviously, it can happen to have the same particle several times and to lose some particles.


## Additional considerations
Even in the implementation we do not really consider orientations directly, if the particle is in the wrong orientation, even if on the current step the observations are OK (they are invariant to rotation), in the next movement, it will move wrong, so it will decrease its importance weight.

We can measure how good our Particle Filter is performing by computing some error between the set of particles and the actual robot pose. One option is the mean euclidean distance in the X and Y positions. Regarding this, we need to consider that the world is cyclic, so if a particle is in X=99.999, it is actually very close to be correct if the robot is in X=0.0.
