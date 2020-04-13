# Trajectory Generation

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

After determinating the high level behaviour we want to carry on, we need to go an step deeper and define which curve we are going to follow and which actions we are going to perform in each timestep to reach our goal.

## Problem Definition
We already know algorithm such as A-Star, Dynamic Programming and Optimal Policy to solve motion planning. Regarding Motion Planning itself, there are several key concepts that we need to know what is the **configuration space**. It is the space that represents all the possible configurations of our robot (possible positions and orientations of the car). We  can then define *motion planing* as a problem in which we receive an start configuration, a goal configuration and some constraints (how the vehicle is able to move, its dynamics, a description of the environment, etc.).

Usually, the localization module and the sensors are the ones who provides us with our starting configuration. The Behaviour Planning Module is what will provide us with the goal we want to achieve. The constraints are mostly defined by physics, the map and the traffic rules.

Our problem itself is to get the sequence of feasible movements in the configuration space that will be able to transform the start configuration to the goal one without hitting any obstacles.


## Types of Planning Algorithms
There are some properties of the Motion Planning Algorithms. These are:

- **Completeness**: If a solution exists, the algorithm will find it.
- **Optimality**: The algorithm will always provide the solution that minimizes a certain cost function.

We can also classify the Planning Algorithm into several types (even we will focus in only one). These types are:

- **Cominatorial Methods**: They divide the space into pieces and solve the motion problem by connecting these pieces. They are very intuitive to get approximations of the optimal solutions, but usually have a bad scallability.
- **Potential Fields or Reacting Methods**: Each obstacle will create a sort of repelling field that will make it harder for our car to move close to it. They can sometimes push us into local minimums.
- **Optimal Control**: They try to solve the Planning problem as well as the Controlling Input Generation. They forward inputs (steering angles, throttle, etc.) through a dynamic vehicle model in order to go from the start configuration to the goal one minimizing an cost associated to the inputs (e.g. minimize gas consumption, maximize distance to other cars, etc.). Most of them are based in numerical optimization methods. Their downside is that it is hard to take everything into consideration while keeping the algorithm real-time.
- **Sampling-Based Methods**: The ones we will be focusing on. They use a *Collision Module* that will identify if a configuration is or not a collision. While the other methods analyze the whole space, not every part of the environment needs to be explored with this type of algorithms. It stores the explored parts in graph structures. It can be subdivided in:
  - **Discrete Methods**: Deterministic. They use a finite set of configurations and/or inputs, such as a grid superposed to our configuration space. Some examples are: A-Star, D-Star, D-Star lite, Dijkstra's or ARA-Star.
  - **Probabilistic Methods**: Rely on a probabilistic sampling of a continuous configuration space. Given that we have infinite(ish) configurations, these methods are (probabilistically) complete and optimal (when allowing enough computation time). They explore the configurations with random exploration. Some probabilistic graph search algorithms are: RRT, RRT-Star, and PRM.


## Hybrid A-Star 
As a quick reminder, the **A-Star** algorithm uses an optimistic heuristic function that helps us to search through a discrete space using a prior. This prior can be, for example, the Euclidean or the Manhattan distances from each cell until the goal. It is complete and optimal (assuming an admissible heuristic and relative to the discretization). Given our expansion options for the cell we are in the stepX, we will expand to the one whose *g* (steps already done) plus *h* (the heuristic function for this cell) is the smallest. The result of this addition is usually called *f*.

Even the traditional A-Star algorithm gives us good solutions and saves us quite a lot of computation time, we may find too sharp turns that are not feasible for a car to do. They key to solve this is in the transition functions. Instead of storing just the cell, we will compute a continuous configuration (x, y, theta) inside this cell and associate the configuration to this cell. This is why hybrid A-star is not purely discrete or continuous. We will use this semi-continuous point to figure out which will be the next cell. This algorithm is not complete (non-considered solutions may exist because we only consider one continuous configuration by discrete cell), but we get smooth trajectories that we will be able to perform with a car.

To sum up, it searches in a continuous configuration space, uses an optimistic heuristic function and its solutions are drivables. However, it is not complete and the solutions may not be the optimal ones.

### Hybrid A-Star in Practice
Regarding the equations to obtain the vehicle's configuration, we should use the bicycle motion model to properly specify the dependency between turning and advancing (cars are not holonomic). We can also define a discrete amount of possible steering angles to make the computation lighter. We should also use a 3rd dimension for the orientation in a way that, in case we close a cell because we arrive to it with an orientation that makes it impossible for us to keep advancing, we will only close this position when we get there with this orientation, because with other orientation we may be able to actually find a way to our goal.

This lesson's [video](https://www.youtube.com/watch?v=Mkz_WjyRzag&feature=emb_logo) illustrates it with some execution examples.

### Hybrid A-Star Heuristics
In [this paper](https://d17h27t6h515a5.cloudfront.net/topher/2017/July/595fe838_junior-the-stanford-entry-in-the-urban-challenge/junior-the-stanford-entry-in-the-urban-challenge.pdf) and, in particular, the section 6.3, goes into detail about how the Stanford team thought about heuristics for their Hybrid A-Star algorithm (which they tended to use in parking lots). It specifies that they use two heuristics. (1) The first one takes into account the non-holonomical nature of a car and assumes no obstacles, which helps with arriving to the goal with the appropiate orientation and can be precomputed. (2) The second one, keeps in mind the obstacles but assumes holonomic movement. This is computed online using Dynamic Programming and makes the car to follow the real shortest path.

Following this idea, if we use two admissible heuristic functions (*h1* and *h2*) and we want to combine them to use it as already explained, we need to make sure that the resulting *meta-heuristic* function is still an optimistic heuristic (i.e. reflecs minor or equal costs to the real ones). In order to do this, we could combine them in several ways, such as *h = (h1+h2)/2, h = min(h1, h2)* or *h = max(h1, h2)*.

NOTE: Hybrid A-Star is implemented in the *00_hybrid_a_star* folder in C++. Originally, it was the *breadth first* algorithm (original files contained in *./00_hybrid_a_star/original_breadth_first/*), which does not use any heuristics. The challenge was to modify it in order to transform it in the Hybrid A-Star. The *breadth first* method uses 12606 expansions and finds a path with 33 steps. The course's solution uses the Manhattan Distance as heuristic function, lowering the expansions to 10082 and finding a path with 38 steps. Some possible improvements:

- Check other distances (Euclidean, etc.) depending on the available movements.
- Fuse with on-line building dynamic programming (such as Junior from Stanford).
- Precompute the whole heuristic function when initializing. NOTE: This may not scale well...


### Taking Advantage of the Environments
We can have **unstructured** environments, such as parkings or mazes, which usually have less specific traffic rules than more **structured** ones (such as highways or streets, where we have speed limits, lane boundaries, traffic direction, etc.) and also lower speeds. A big characteristic of these environments is the absence of a clear driving path to follow. Hybrid A-Star is one of the bests approaches to find paths everywhere. However, it does not take advantage of the extra information of structured environments, so it is more outstanding in the more unstructured ones.

## Further Considerations
Whenever we want to use the structuration of an environment such as a road, we may want to use **Frenet Coordinates** to model our position in it in a more intuitive way (*[s, d]*, the longitudinal and lateral position on the lane).

Another important concept is to keep track not only of the positions, velocities and accelerations of the cars, but also of the time in which each of them occured. One trajectory may seem exactly as other one but can mean very different things if one was at a constant speed and the other involved slowing down (or even stops) and accelerations. This is why driving is a 3D problem: *s, d* and *t*. We need to know where the other objects will be but also when they are going to be there in order to avoid collisions during our trajectories.

So far, we have managed to create paths when having static obstacles around us. However, if they move or a new obstacle appears while executing our planned trajectory, we may have a collision. The solution to this is predicting where the other obstacles will be in each time (with our Prediction Module) and assign each point of the trajectory a time in which will happen. This way, our trajectories will contain configurations of the car (location (+ orientation)) and the time in which it should be reached. In order to visualize this, we usually build two graphs, one with time vs *s* and other one with time vs *d*. This way, we can visualize changes on velocity and on lanes, respectively, in a more clear way.


## Sampling-Based Polynomial Trajectory Generation
This method for Trajectory Planning is very useful for structured environments such as highways or streets. It uses boundary conditions and jerk minimization to generate polynomial trajectories. We will study what is the *jerk*, how to derive the polynomial's parameters to solve our problem and how to evaluate the feasibility and cost of the trajectories. We will also see how to generate several similar trajectories that minimize the *jerk* and compare them to chose the best one.

### Jerk-Minimizing Trajectories
We can not use linear conexions between two points of our trajectory (in the *s-t* or *d-t* spaces suggested before) because they would mean instantaneous position or velocity changes that are not feasible and, even if we forward these commands to our engine control, it will create very high accelerations, which is dangerous and uncomfortable. In order to avoid this, we want our acceleration to be continuous, and maybe its derivative (***jerk***) aswell. The derivatives are: position-velocity-acceleration-jerk-snap-crackle-pop. Humans have no problem with traveling at high velocities (e.g. airplanes), but we start to be sensitive to acceleration, even we can be OK with the ones under 1G (the maximum a normal car can sustain). The real problem comes when the *jerk* is high. A high jerk implies a quick change in acceleration (such as in bump cars) and that is what we perceive as uncomfortable and it is also dangerous, that is why we want to use **jerk minimizing trajectories**.

For 1D scenarios, minimizing the jerk is fairly easy. We just need to use a 5th order polynomial, as explained in the [video of this lesson](https://www.youtube.com/watch?v=pomDFkzy2bk). This means that we will have 6 tunable parameters that we can use to specify the boundaries of our trajectory, such as initial and final position, velocity and acceleration. We can use this for both longitudinal and lateral displacements (*[s_i, s_dot_i, s_dot_dot_i, s_f, s_dot_f, s_dot_dot_f]* and *[d_i, d_dot_i, d_dot_dot_i, d_f, d_dot_f, d_dot_dot_f]*). Using this, we will be able to use 12 variables in order to define the motion of our vehicle in *s* and *d* (of the Frenet Coordinates) over time. Given our 6 boundaries, we will obtain the 6 parameters derivating the 5th order polynomial twice, substituting and solving.

It is relevant to highlight that, even this strategy will give us the smoothest possible trajectory given our restrictions, it will depend a lot on them. If we ask the car to reach a certain configuration in a very small amount of time, the minimum required acceleration will still be large.

A ***polynomial solver*** will take as its inputs the current vehicle state (pos, vel and acc), the goal state and the duration of the maneuver. Then, it will process them to output the 6 coefficients that define the polynomial that define the jerk-minimizing trajectorie to follow. We will do this for both our tangential (*s*) and lateral (*d*) configurations. This is implemented in C++ in the *main.cpp* inside the *01_quintic_polynomial_solver* folder.

### Feasibility Evaluation
When evaluating how feasible a trajectory is, we need to take into account its maximum velocity (to check it is possible for the car to reach it and that it does not exceed the speed limits), its minimum one (may be risky to go too slow or going fast backwards), its maximum and minimium accelerations (to avoid slowing down or aumenting the velocity too fast) and the steering angle (to avoid abrupt turns and make sure they are possible for the car).

To check the feasibility of a trajectory, we can assume that the road is locally straight. We will also assume that our heading is the same than the one of the road. Now we need to check that:

- The longitudinal acceleration is less than the maximum than the engine can supply and more than the maximum breaking acceleration of the car. This could be fixed values or computed given the current road friction.
- The lateral acceleration should be less than a fixed value, determined for comfort and rollover risk avoidance.
- We can use the bicycle model to determine the maximum steering angle both clock and counter-clockwise.
- We will make sure that the longitudinal velocity is under the speed limit and over the minimum one we want to have (e.g. for safety reasons in a highway or to avoid backing up).

### Integration With the Behaviour Planning Module - Cost Analysis
Usually, our Behaviour Planning will not output an specific configuration, but an approximation of one. Here is where the **Sampling-Based Approach** becomes handy. Since we do not know which is the best goal state, we can sample a large number of configurations, generate trajectories, remove the invalid ones (non-drivable, collisions, uncomfortable, etc.)  and compare them to chosee the best one.

NOTE: Even if we receive a precise *s* and *d* coordinates (which is not usual), we will still have to chose a velocity and an acceleration.

In order to compare each trajectory, we need to rank them using a cost function. We can consider the following aspects:

- Even the jerks are minimal for every trajectory, different trajectories will have a bigger or smaller jerk, so we can use this as a measure.
- The lateral jerk is more uncomfortable than the longitudinal one.
- We will also prefer to be as far away from the other objects as possible.
- It is more convinient to be as centered in the lane as possible.
- Finally, we want to arrive to our goal as fast as possible, which means sooner in the time dimension. 

We could think about many more cost functions. The tricky part is to fuse all of them correctly using the best possible weights. Often they will find conflicts, such as when the one with the smalles jerk is the slowest one.


## Polynomial Playground 
In the *02_polynomial_playground* folder there is a working Polynomial Trajectory Generator (PTG). The weights of the cost functions are not properly configured, so the simulated car is not able to perform the intended action (get behind a target vehicle). The challenge is to solve that. In the 5 provided files, it might be convinient to start by tuning the weights in *ptg.py*, but we could also add more cost functions. As an extra challenge, it is suggested to try to implement this in C++. The provided files are: 

- **ptg.py** - The primary code for generating a polynomial trajectory for some constraints. This is also where weights are assigned to cost functions. Adjusting these weights (and possibly adding new cost functions), can have a big effect on vehicle behavior.
- **cost_functions.py** - This file contains many cost functions which are used in ptg.py when selecting the best trajectory. Some cost functions aren't yet implemented...
- **evaluate_ptg.py** - This file sets a start state, goal, and traffic conditions and runs the PTG code. Feel free to modify the goal, add traffic, etc... to test your vehicle's trajectory generation ability.
- **constants.py** - constants like speed limit, vehicle size, etc...
- **helpers.py** - helper functions used by other files.

The files can be downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/December/5a3ea459_trajectoryexercise2-python3/trajectoryexercise2-python3.zip) or [here](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/October/59d5bfcb_trajectoryexercise2/trajectoryexercise2.zip). The ones in the folder are from the first link.


## Further Reading
- A Polynomial Trajectory Generation [paper](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/July/595fd482_werling-optimal-trajectory-generation-for-dynamic-street-scenarios-in-a-frenet-frame/werling-optimal-trajectory-generation-for-dynamic-street-scenarios-in-a-frenet-frame.pdf) discussing some interesting (and potentially useful) topics like: cost functions, differences between high and low speed trajectory generation, specific highway maneuvers implementation, lateral and longitudinal trajectory combination and derivate from Frenet to Global Coordinates.
- Indoors Path Planning: [Intention-Net: Integrating Planning and Deep Learning for Goal-Directed Autonomous Navigation](https://arxiv.org/abs/1710.05627), by S. W. Gao, et. al..
- City Navigation: [Learning to Navigate in Cities Without a Map](https://arxiv.org/abs/1804.00168), by P. Mirowski, et. al.
- Intersections Path Planning: [A Look at Motion Planning for Autonomous Vehicles at an Intersection](https://arxiv.org/abs/1806.07834), by S. Krishnan, et. al.
- Traffic Path Planning with Deep Reinforcement Learning: [DeepTraffic: Crowdsourced Hyperparameter Tuning of Deep Reinforcement Learning Systems for Multi-Agent Dense Traffic Navigation](https://arxiv.org/abs/1801.02805), by L. Fridman, J. Terwilliger and B. Jenik.

