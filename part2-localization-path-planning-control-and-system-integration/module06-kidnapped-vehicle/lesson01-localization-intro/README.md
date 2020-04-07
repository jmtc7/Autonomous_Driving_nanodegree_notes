# Introduction to localization

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The idea of localization consists in gathering information about our environment and compare it to information we already have to try to know our position. That is what we would do if we were kidnapped and, after several hours of blindfolded travel, we are unblindfolded and asker where are we.



## Sensors and maps
Usually, the required precision for localization problems when working with cars is of around 3 and 10 cm. GPS is a good source of information, but it gives us an accuracy of 1 to 3 meters and it can grow up to 50 m. That is why we use LIDARs and RADARs to measure distances until trees, walls, etc. When these obstacles from our local environment are also part of the map we have, we can use these measurements to preciselly localize the car in the map.

In order to use the measures with the map, we need to match them in order to compute a tranformation between our local coordinate system and the global one of the map.


## Module structure
We will start by an introduction and a simple Python implementation. The next step will be the Bayesian Filters and their 1D implementation in C++. Next will come the motion models to finish with the Particle Filters and its 2D implementation in C++.


## The localization problem as a probability distribution
If we have 5 different positions in which a robot can be, without any other knowledge, each of them will have a 1/5 = 0.2% of probability to be the actual robot position (the probablity should add 1). If 3 cells are green and 2 red and the robot makes an observation of the floor and it sees it red, we can multiply the red cells' probabilities by a factor (e.g. 0.6) and the green ones by another (e.g. 0.2). This will result in a higher probabilites for the red cells rather than for the green ones. However, now our probability adds 0.36, not 1, so we need **normalization**. We can do this by dividing every number by the un-normalized sum of the whole collection (0.004/0.36=1/9 and 0.12/0.36=1/3). These results will be the *posterior distribution*, i.e. the probability of the position given an observation (*P(x|z)*).

We can do this several times using a sequence of perceptions, as is done in the *01_multiple_measurements.py* script.
