# Markov Localization

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this lesson, we will go more in depth into localization, studying the derivation of the Bayes Localization Filter, which is called ***Markov Localization***. In parallel, we will be implementing it for a 1D localization problem in C++.

## Formal Definition of Variables
Through this and the following lessons, we will see the measurements as the vector *z*, which will contain the measurement from the instant *1* until the intstant *t* (current or final time). The control commands will be the vector *u*, again, from *1* to *t*. The global map will be *m*. It could be a grid map or a collection of features and it will be constant in time, that is why it does not go from *1* to *t*. 

The target is to compute the transformation between the local coordenate system of the car and the global one of the map. The car's pose in the global map's coordinates will be *xt*, which will contain the position and orientation. This will never be perfectly known. We will use a belief *bel(xt)* on this pose. This belief will be the car's pose given the sequence of observations and movements and the map. i.e. *bel(xt) = P(xt | z1t, u1t, m)*.

NOTE: If we don't assume a constant, known map, we will be solving the Simultaneous Localisation and Mapping (SLAM) problem, and we will be trying to increase *P(xt, m | z1t, u1t)*.


For this lesson, we will be considering a 1D localisation problem. The map will be a feature map consisting in a list of the distances of the features to the center of the map coordinate system. (e.g. *[9, 15, 25, 31, 59, 77]* indicating the position of trees or street lamps). Each observation in the observation list (*z1:t*) will be a vector composed by all the detections perceived at this timestamp. The car's movement history will be a sequence of the ammount of meters it advanced in each timestamp. We will assume a discrete map and limit it between 0 and 99, so the car's pose will be any integer in this interval. Therefore, the belief on the position of the car will be a vector of a 100 elements, each of which will contain the probability of the car being on it.


## Bayes Filter for Localization
The bayes theorem applied to the localization problem will be defined as follows: *P(X_t|Z_1:t) = (P(Z_1:t|X_t) x P(X_t))/P(Z_1:t)*. Note that *P(X_t)* is determined by the motion model. It is the product of the transition model probability (probability of moving from *X_t-1* to *X_t*) and the probability of the state *X_t-1*.

In the next quizes, I will code the following things:

- Compute Bayes’ rule
- Calculate Bayes' posterior for localization
- Initialize a prior belief state
- Create a function to initialize a prior belief state given landmarks and assumptions



## Bayes Rule with Additional Conditions
If we are working with a LIDAR providing 100000 data points each of them containing 5 pieces of information (point ID, range, 2 angles and reflectivity) and each of them requires 4 bytes of memory, we drive for 6 hours and the LIDAR gives 10 full scans each second, this will mean that we will gather 432 GB of data. 

This would be the amount of data that we will have to take into account in our *Z_1:t* vector if we change nothing on the current approach. This is unfeasible because of the huge amount of data and its increase over time because we would not be able to do all the calculations in Real Time. Next, we will see how we can do this using only a few bytes of data and to keep this quantity of data constant over time.

If we manage to make the Bayes Filter a recursive function in a way that the current belief could be computed using only the last belief and the new observations, we would solve this problem. That is what we call the ***Bayes Localization Filter*** or ***Markov Localization***. In order to do this, we need to use:

- Bayes Rule
- Law of Total Probability
- *Markov Assumptions* about the dependency between certain values

We do this by using our posterior belief (*P(x_t|z_1:t, u_1:t, m)*), separating the current observation from the previous one (*P(x_t|z_t, z_1:t-1, u_1:t, m)*) and considering *x_t* our *A*, and *z_t* our *B*, keeping all the other consideration always on the right of the assumption of the Bayes Rule, getting the following expresion: *P(x_t|z_t, z_1:t-1, u_1:t, m) = (P(z_t|x_t, z_1:t-1, u_1:t, m) x P(x_t|z_1:t-1, u_1:t, m)) / P(z_t, z_1:t-1, u_1:t, m)*.

We call the likelihood the ***observation model***, which is *P(z_t|x_t, z_1:t-1, u_1:t, m)*. Here we are assuming that we know the previous state, measurements, motion commands and map.

The prior is called the ***motion model***: *P(x_t|z_1:t-1, u_1:t, m)*. No current observations are included.

Finally, to simplify the normalization, we will define our normalizer as the constant *Eta*, which will be *1/P(z_t, z_1:t-1, u_1:t, m)*. By the Law of Total Probability, it will be equivalent to the sum of the product of the observation and the motion models over all the possible states (all the possibles *x_t*).


## Motion Model
Regarding the **Markov Assumptions**, the first one says that we can use only the last position to know the actual one because it would contain the information (and uncertainty) of all the past ones. i.e. to process *x_t* we would only use *x_t-1* because it will contain the information of *x_t-2*, *x_t-3*, etc. This means that we are assuming that the poses are only dependent to the ones just before them. In our motion model (*P(x_t|z_1:t-1, u_1:t, m)*), we can apply this because *z_t-1* and *u_t-1* will affect *x_t-1*, so it will carry all the necinformation and transmit it to *x_t*. This way, while working with *x_t*, we only need to consider *x_t-1, z_t, u_t* and *m*. This will simplify our **transition model**, which will now be *P(x_t|x_t-1, u_t, m)*. This is a part of the Law of Total Probability with which we can extend our motion model from *P(x_t|z_1:t-1, u_1:t, m)* to *SUM(P(x_t|x_t-1, z_1:t-1, u_1:t,  m) x P(x_t-1|z_1:t-1, u_1:t, m))*. This, using the 1st markov assumption, will end up condensing all the info in the previous state so that we can simplify this as previously explained.

If we apply the Markov assumption to the second term (*P(x_t-1|z_1:t-1, u_1:t, m)*) of the Total Probability-expanded motion model, we can not consider *u_t* because it happened on the future (with respect to *x_t-1*). This way, we end up with this new second term: *P(x_t-1|z_1:t-1, u_1:t-1, m)*, which lead us to this new expresion of the motion model: *P(x_t|z_1:t-1, u_1:t, m) = UM(P(x_t|x_t-1, z_1:t-1, u_1:t, m) x P(x_t-1|z_1:t-1, u_1:t-1, m))*, which is a **RECURSIVE STRUCTURE** that allows us to solve the data problem!! 

NOTE: Many approaches also consider that the state is independent from the map, so they remove this term aswell, but we will keep it.

NOTE 2: All this, following the chain, ends in giving a very high importance to the first estimated state, which can't be estimated using any previous state. To initialize it in a decent way we can, for example, use the GPS localization, whose high error will be reduced as we keep performing new observations with our other sensors.


This recursive version is implemented in the *02_discrete_recursive_motion_model* folder in C++.


## Observation Model
This defines the probability of the current observation given the current state, the collection of previous observations, all the motion orders and the map (i.e. *P(z_t|x_t, z_1:t-1, u_1:t, m)*). We will use the *Markov's Assumption* again to simplify this expression. Since we are considerating *x_t*, we don't care about the observations or movements from the past because they are already considered in *x_t*, so we can assume *z_t* to be independent of these things, which leaves us with *P(z_t|x_t, m)*. This tells us that every observation is independent of each other, so it can be the product of all the individual observation probabilities.

The observation models will change depending on which sensor and map are using. For this project, we will assume a sensor that will provide the measurements to the landmarks in the 100m in front of the car.

In the *03_get_pseudo_ranges* folder, a function that generates the *pseudo-ranges* (expected measurements, i.e. landmark position on the map - car position on the map) is implemented. The whole observation model probability computation is implemented in the files inside the *04_observation_model* folder.


## Bayer Filter Theory Summary
The Bayes Localization Filter is a general framework for **recursive state estimation**. *Recursive* meaning that we will only need the current observation and controls, without needing the whole history of data. This has two parts: (1) the **motion model** (equivalent to a *prediction step*) and (2) the **observation model** (equivalent to an *update step*). This prediction and observation loop is the idea behind Kalman Filters, as well as the studied 1D localization and the Particle Filter that will be studied in the next lessons. This means that all of these techniques are realizations of the Bayes Filter.

In the *05_full_bayes_filter* folder, a full implementation of a Bayes Filter for localization can be found. It uses the Markov's Assumptions for making it feasible and includes prior initialization, motion model, pseudo-ranges generation, observation model, normalization and prior updating.

In the next lessons, we will see more advanced concepts such as complex motion models and Particle Filters for 2D localization.


