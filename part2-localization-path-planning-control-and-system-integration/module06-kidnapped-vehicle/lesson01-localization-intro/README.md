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


## Robot motion
In the case of a perfect exact motion model, if the robot moves one cell to the right, we would shift all the probabilities to the right. However, we usually have some inaccuracy in the movements we do, so there is a chance the robot will end in the previous or the next cell to the target one. Since we can arrive to whichever cell from whichever other, we can apply the motion model (and its uncertainty) as a convolution.

If we execute infinite times a certain motion, we will reach the ***limit distribution*** regarding the position probability. This is usually a flat distribution in which the robot can be anywhere because the uncertainty has increased and increased along the process. This is related to the ***balance property***. The probabilities that add the probability of other cell, need to be balanced so that their relationship and dependance is still true in the limit distribution. All these cycles are implemented in the *01_multiple_movements.py* script.


## Observation-motion cycle
This is the cycle in which localization consist. We increase the uncertainty about our location when moving and icrease it thanks to observations. We can *measure* how much information we have using the **entropy**. 

We can call the probabilities *belief* and the perception or *sense* will be equivalent to a product (the x0.6 or x0.2 we did) followed by a normalization. The *moves* will be equivalent to a convolution and an addition per each cell. A sequence of observations and movements is implemented in the *03_sense_and_move.py* script.


## Bayes Rule
We always have to keep in mind that our probabilities have to add one.

Bayes Rule is the most fundamental consideration in probabilistic inference. If *X* is the position in a certain grid cell and *Z* our measurement, the probability of being in a certain cell after having perceived a certain observation is P(Xi|Zi). The Bayes rule says that P(Xi|Z) = (P(Z|Xi) x P(Xi))/P(Z), where *P(Xi)* is the prior distribution, *P(Xi|Z)* are the chances of seing what we have seen from our current location in the map (measurement probability, which was large when the measurement corresponded to the correct color and small if it did not). Its outcome is the non-normalized posterior distribution *P(Xi|Z)*.

This product (*P(Z|Xi) x P(Xi)*) is what we programmed before, the probability multiplied by the *pHit* or *pMiss*. Finally, the normalization is done by the constant *P(Z)*. Notice that the *Z* does not have an *i*. This is because it corresponds to the sum of observing what we saw from every possible location *P(Z) = sum(P(Z|Xi) x P(Xi)*.

Given our motion problem, the probability of being in the *i* cell in the *t* time is: *P(Xi,t) = sum(P(Xj,t-1) x P(Xi, Xj))*. i.e. the sum of the probabilities of every grid cell (*j*) in the previous moment (*t-1*) multiplied by the probability of the robot ending up in the cell that is being evaluated (*i*). This is what we coded before (suming the probabilities of being in a certain cell taking into account exact, overshooted and undershooted movements). This is known as the ***Theorem of Total Probability*** (*P(A) = sum(P(A|B) x P(B))*) and is what we used to define *P(Z)* in the last paragraph. This *weighted sum operation* is called a ***convolution***.


## Additional resources
Some interesting resources related with Bayesian Methods are:

- [Sebastian Discusses Bayes Rule](https://classroom.udacity.com/nanodegrees/nd013/parts/30260907-68c1-4f24-b793-89c0c2a0ad32/modules/28233e55-d2e8-4071-8810-e83d96b5b092/lessons/3c8dae65-878d-4bee-8c83-70e39d3b96e0/concepts/487221690923?contentVersion=2.0.0&contentLocale=en-us)
- [More Bayes Rule Content from Udacity](https://classroom.udacity.com/courses/st101/lessons/48703346/concepts/483698470923)
- [Bayes Rule with Ratios](https://betterexplained.com/articles/understanding-bayes-theorem-with-ratios)
- [A Deep Dive into Bayesian Methods, for Programmers](http://greenteapress.com/wp/think-bayes)
