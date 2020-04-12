# Prediction in Path Planning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The predictions are inherently multimodal, given that we will have several peaks on our probability distribution. e.g. a car approaching to an intersection is more likely to keep going straight, turn left or turn right, so we will have 3 peaks in the distribution.

We usually keep track of the probabilities that our tracked object will do each of the actions it is most likely to do. We can use some prior as statistican knowledge that in the given intersection, cars usually do not turn. However, if in the following time steps we see the car slowing down, we may want to increase the probability related to the car turning and decrease the one of going straight. If we perceive changes on the car orientation or detect the blinkers activated, we will keep wanting to increase this probability.

In order to do this, we usually forward a map of the environment and data from sensor fusion to our prediction module to get a prediction on what another object will do in order to decide if our next action is safe enaugh. Regarding the **prediction module**, we have model-based and data-driven approaches, depending if it is based on mathematical models or in machine learning.

## Sample Input/Output
In the files *input.json* and *output.json*, some sample inputs to the prediction module and the generated outputs can be appreciated. In the input, the perception/data fusion have provided us with 2 cars and their position, velocity and uncertainty. The output consists in some higher level information, such as the shape of the cars, and predictions on their position and orientation in the near future.

In reality, we will make predictions for the next 10-20 seconds with a resolution of over 0.5 seconds. Moreover, we will be tracking all the dynamics objects around us, not only cars.


## Types of Approaches
The two main styles on prediction are **model-based** and **data-driven**. The first one uses mathematical models to estimate where an object will go and will try to match these predictions with observations. Finally, it will decide which is being the followed trajectory given observed behaviour and a **Multi-Modal Estimation algorithm**, which will be a black box for now.

The **data-driven** approaches use Machine Learning to, after being trained, receive the observed behaviour and directly estimate a probability of where the target is going.

### Comparison
The advantage of the model-based approaches is that they use our human knowledge (physics, road rules, etc.) directly. However, the data-driven ones may be able to detect subtle indicators that we may not have been able to include in our mathematical models.

Each of them may be more convinient for ones or other situations. For example, in wet conditions, the model-based ones will be way more convinient to be able to model the friction and forces very precisely. However, if we find an unrecognised object in the road, the data-driven approach will be able to generalise what it learned for other objects, while the model-based one will not be able to assign any model to it.

For general situations, both styles will be able to succeed. However, we should keep in mind that there is always the posibility of using a hybrid technique.

### Data-Driven Example: [Trajectory Clusteringi](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/July/5978c2c6_trajectory-clustering/trajectory-clustering.pdf)
It will require a previous training phase before it is able to be used to generate online predictions. Regarding the **offline training**, we could gather a lot of data placing cameras in intersections. We will probably have to clean the data. Next, we will require a mathematical expresion to model the similarity between the trajectories. Once we have defined how to know if two trajectories are or not similar, we can perform Machine Learning using **unsupervised (agglomerative or spectral) clustering**. In a 4-road intersection, we will expect to see 12 clusters (4 roads, in which we can do 3 actions: (1) turn left or (2) right or (3) keep straight). We may get extra clusters, as if we have a traffic light instead of a stop, in which we may get 3 clusters for the cars that go trough the intersection without stopping and other 3 for the cars that have to stop (6 in total for each road). Once having the clusters, we will define prototypical trajectories for each clusters (e.g. we may want to use 3 models for each cluster).

For the **online usage** of this technique, we will observe other vehicle's partial trajectory and we will compare it with one of our prototypical trajectories of each cluster. We will reutilize the similarity measurement used in the training. The belief on each trajectory will be proportional to the similarity between the partial trajectory and the prototypical trajectories. Finally, we will take the most similar prototypical trajectory of each cluster and evaluate the probabilities that the car is following each one of them. The one with the higher probability will be the one that we will consider.


## Model-Based Approaches
### Frenet Coordinates
These are a more intuitive way to represent the position inside a road than the traditional *[x, y, theta]* option. This consist in using *s*, that will be the longitudinal displacement across the road, and *d*, the lateral displacement relative to the road center.

### Process models
While data-driven approaches rely just in past evidence to predict future behaviour, model-based ones may give us more certainty because they allow us to use deterministic knowledge about physics, vehicle dynamics, road rules, etc. We usually use them as follows:

- For each dynamic object, identify its possible behaviours (lane changing or turns for cars, cross the street for pedestrians, etc.).
- Describe a process model for each behaviour. This is a mathematical description of the motion. It should incorporate some uncertainty.
- Update each model's uncertainty comparing the new observations with the process model's output.
- We will end up chosing the most likely trajectory and using it as our prediction.

Some process models are:
- **Linear point model**: Assumes a holonomic vehicle represented as a point in the space with constant velocity.
- **Non-linear point model**: Considers constant acceleration and allows considering the heading of the car.
- **Kinematic bicycle model with controller**: Stops considering the car a holonomic vehicle and uses a PID controller to correct the steering angle so that the car will drive in the center of the lane while respecting the speed limit.
- **Dynamic bicycle model with controller**: Usually too complex. It takes into account the lateral forces on the front and rear of the vehicle. Usually, due to the other driver's behaviour prediction uncertainty, the small increase of precision that this model offers usually does not make sense.


In the sections 3.1 and 3.2 of the paper [*A comparative study of multiple-model algorithms for maneuvering target tracking*](https://d17h27t6h515a5.cloudfront.net/topher/2017/June/5953fc34_a-comparative-study-of-multiple-model-algorithms-for-maneuvering-target-tracking/a-comparative-study-of-multiple-model-algorithms-for-maneuvering-target-tracking.pdf), it is explained how we can combine several process models to predict behaviours. In particular, this paper proposes one constant velocity model (CV), four constant tangential acceleration ones (CTA) and four constant turn models (CT). Each CTA model is configured with different accelerations (+-33 and +-60 m/s2) and each CT one, with different turns (+-7 and +-14 deg). This way, we will be tracking the objects assuming that they will be following one of the 9 trajectories modeled by these process models. Each model has a certain ammount of noise associated (2.25 m/s2 for CV and 50 m/s2 for all the maneuver models).

### Multi-Modal Estimators
In order to maintain some belief about what a driver will do, we use the Multi-modal Estimation algorithms. A simple approach to that is the **Autonomous Multiple Model Estimation** (AMM). We will have *M* models, each of them with a *mu* probability (*mu_0, mu_1*, ..., *mu_M*). Given, for example, two models and an observation in time *k-1* and another in time *k*, we will compute the probability of each behaviour in each time step (*mu_k*) taking into account the last probability (the one in *k-1*, i.e. *mu_k-1*) and how similar is the new observation to the state that each model predicted for the time *k* (i.e. the likelihood *L_k*). The probabilities are normalized.
Multi-modal estimators (to handle the prediction-associated uncertainty)


## Hybrid Approaches
They combine both data-driven approaches and model-based ones to benefit from each one's advantages. To do so, instead of combining process models with a multi-model estimator, we combine them with a Machine Learning-based classifier, such as the **Naive Bayes** classifier.

### Naive Bayes Classifier
This allows us to know the probability of something given some additional information, assuming that this *additional information* contributes independently. As an example, we could compute the probability of being male or female given the height and the weight. According to Naive Bayes, it will be something like:

```
P(male | w, h) = (P(h | w, male) * P(w | male) * P(male)) / P(w, h)
```

This is, the probability of a male having this height multiplied by the probability of a male having this weight multiplied by the probability of being male (~0.5). All of it will be divided by the overall probability of an individual having this weight and height. Since this last term will affect both men and women, we could remove it and normalize using the sum of the probabilities (if we do not know it).

Therefore, we will have the problem finding *P(h | w, male)* and *P(w | male)*. Often, we can assume that they are in a gaussian distribution for the feature variables. i.e. both males and females will have a mean height with a certain variance and same for the weight. If we use this approach, we will be using the **Gaussian Naive Bayes**. This algorithm will have the next steps:

- Select the relevant features. We can use human intuition and/or feature selection algorithms. e.g. the eye color is not as meaningful as height to know a human gender.
- Define gaussian distributions (means and variances) for each feature for each class. We can do it by guessing or by gathering a lot of data.

In the T-shaped intersection, we could define which features are relevant to know if a driver is going to keep going straight or to turn (such as blinkers, velocity reduction, orientation variation, etc.) and use Naive Bayes to compute the probability of each process model at each time step. We would be using the model-based process models and the data-driven feature probabilities.

### Naive Bayes C++ Implementation for a 3-lane Highway

This is implemented in C++ inside the *00_naive_bayes* folder. It considers a 4-meter width highway with 3 lanes. For the center lane, the cars will be able to keep going straight, to change lange to de right or to do so to the left. For the top and bottom lanes, they will only be able to go straight or to swith lane to one of the sides.

We will be using Frenet Coordinates and the target will be to determine which trajectory/action a driver is doing given its *s, d, s_dot* and *d_dot*. To do so, we will implement the *GNB::train()* and the *GNB::predict()* methods in *classifier.cpp* for the Gaussian Naive Bayes algorithm.

Some helpful resources were:

- [SciKit-Learn documentation on GaussianNB](http://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes).
- [Wikipedia article on Naive Bayes/GNB](https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes).
- [*Load_State()* function issue](https://github.com/udacity/sdc-issue-reports/issues/914). It is not loading data properly from train_states.txt and test_states.txt since istringstream is not implemented for comma (',') separated data.
- Supporting materials (already in the *./00_naive_bayes/support* folder):
  - [Nd013 pred data](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/July/59695c4b_nd013-pred-data/nd013-pred-data.zip). The data for this challenge.
  - [python_extra_practice](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/July/59750f3d_predictionexercise/predictionexercise.zip). It is the same challenge but in Python.
  - [python_solution](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/July/59751005_predicition-solution/predicition-solution.zip). The solution for this challenge (in Python).

NOTE: Improvement suggestion: Build left, keep and right mean, standard variations (variances) and priors in 3 vectors with 3 elements each (instead of the current 9 independent elements of the current version.


