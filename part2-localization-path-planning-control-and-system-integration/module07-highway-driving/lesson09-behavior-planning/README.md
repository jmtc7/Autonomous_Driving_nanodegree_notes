# Behaviour Planning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lesson we will be using finite state machines to determine (at a high level) what to do next given all the data and predictions our system has already gathered. We will also study the cost functions that will let us decide which behaviour is more appropiate.

## Introduction
With very few specificatons we can define a lot of behaviours. For example, in a 3-laned highway, we could use a *target lane, target leading vehicle* and *target speed* to define behaviours like *keep following the car you are following in your current lane*, *change lane and follow "this" vehicle* (to pass the current leading car), *stop following the leading car and follow "this" target speed* (if we are going too fast) or *follow "this" car of "this" other lane and switch to its lane* (match its speed before changing lane).

The responsabilities of the behaviour planning module are suggest *states* (maneuvers) which are feasible, safe, legal and efficient. However, the collision avoidance and execution details are not its concern.


## Finite State Machines for Behaviour Planning
They define transitions between a collection of finite states. They start in a certain state when initialized and the states will be connected by *transitions*. We can have *self-transitions*, that are transitions from one state to itself. There is a chance we will have *accepting states*, which are states that does not have transitions to any other one. The Finite State Machines (FSMs) receive some kind of input and use a *transition function* to decide which will be the next state given the current (and maybe others in time) state and the input.

### Strenghts and Weaknesses
Their **strengths** include:

- Its *self-documenting* nature, because their logical/digital states correspond directly to what they are doing physically.
- Its expandability. We can easilly add new states and transitions to the current scheme.

Regarding their **weaknesses**, we have:

- We can forget to include something in our designs.
- They are prone to be fixed/expanded with patches that can accumulate and become a sloppy and dirty solution.
- They get very complex to expand when they are big because we may forget transitions or restrictions while adding new states.

### FSMs in Self-Driving Cars
The possible states we may encounter while driving in a highway are:

- Keep driving in our current lane.
- Change to the left lane.
- Change to the right lane.
- Follow vehicle in front of us.
- Pass the vehicle in front of us (maybe we can do a change to left followed by a change to right).
- Slow down (if we are following a car that is too fast or want to wait for a gap to change lane).
- Prepare change to left lane (more specific than slowing down if we want to wait for a gap in the left lane, switch blinkers on...).
- Prepare change to right lane (more specific than slowing down if we want to wait for a gap in the right lane, switch blinkers on...).
- Accelerate.
- Stop (for emercency situations).
- Keep target speed (we may include the *slow down*, *accelerate* and *stop* here, but we might define our speed given the speed limit and the car in front of us aswell).

We have many options that may be or not be fully necessary. It is up to us to select how many states we need. We want to pick the minimum amount of them to keep things clean and easy to maintain, but we must include as many states as we need to reach the required performance and safety. From the listed ones, I would personally pick:

- Follow vehicle.
- Change to left lane.
- Change to right lane.
- Keep target speed.
- Prepare for lane change to the left.
- Prepare for lane change to the right.

The instructors selected *keep lane, change lane left* and *change lane right* as the essential ones. They also include *prepare change left* and *prepare change right* to make it easiere to switch on the signals, adapt our speed, wait for a slot, etc. These are, therefore, the states that we will use during the lesson. These are, therefore, the states that we will use during the lesson.


## Cost Functions for Behaviour Planning
Once having our FSM defined (states and transitions), we can assign some weights to each transition. This will allow us to quantify how *expensive* (risky, inefficient, unfeasible or illegal) is a certain sequence of actions. Using these costs we can create rough sequences of transtions and states (hypothetical behaviours) and compare them to decide which of them is *cheaper*. 

### Examples and Implementations
An example one will be penalize large *delta_d* (distances to the center of the target lane) and small *delta_s* (the smaller it is, the more urgent it is to be closer to our goal). In order for the cost to be between 0 and 1, we can use *cost = 1 - exp(-abs(delta_d)/delta_s)*. A similar version of this is implemented in the *00_cost_function* folder, where the function takes into account that both the intended and the final lane are close to the goal lane. It keeps wanting the car to be in the goal lane as soon as possible.

It is common tu use several cost functions in order to archive accurate measurements for complex behaviours. Another relevant parameter that we care about is the time it takes for us to get to our goal. Therefore, a cost penalizing slow lanes will be useful. This is the one implemented in the *01_second_cost_function* folder.

### Cost Functions Design
The usage of these functions introduce several dificulties, such as solving problems without breaking anything that was working before. This can happen when we introduce a new cost function to solve or fine-tune a certain car behaviour but we make something that was working before, stop working. We can solve this by ***Regression Testing***. This consist in defining a certain set of situations in which we will test all new additions to our development.

Another challenge is to balance the costs in different situations. e.g. In a general case we will want to be as efficient as possible, while in an emergency situation, safety is the highest priority, so having low efficiency should never prevent us from performing the safetiest action. We can do this by having **weights** that will prioritize each cost function in order of importance. The most common approach is: Feasibility >> Safety >> Legality >> Confort >> Efficiency. These weights may be **dynamic**. In an intersection, the *legality* concerns about a traffic light turning red are way more important than the normal legality concers in a highway environment because it can put us in very risky situations that our sensors may not be able to detect soon enough.

Finally, reasoning about individual **cost functions** is already a big difficulty. Each of them should serve a very specific purpose. e.g. Our *second cost function*, meant to keep the car driving in the fastest lane, mixed *not to exceed the speed limit* (legality) with *drive as fast as possible* (efficiency), which should not be done. We usually look to have the cost functions' **outputs standarized** between -1 and 1. It is usually beneficial to have tweakable **parameters** in our cost functions to be able to use automatic tuning techniques such as gradient descent to optimize the system's behaviour. It can be helpful to think in terms of the **vehicle state**. i.e. Focus our cost functions in position, velocity and acceleration (the things we can control) and how these aspects can influence in efficiency, safety, etc.


## Further Considerations
Some modules of the autonomous cars will complete full cycles way faster than others (sensor fusion is faster than localization and trajectory planning, which are faster than prediction, which is faster than behaviour planning). This makes it possible to find ourselves in a situation in which we have finished our planning while we have not finished our prediction step, for example. In such situations, we can wait until we have new information of all the modules we use as input or use old information. We should **use old information** because, otherwise, we will be blocking the pipeline.

The **final challenge** of this lesson consists in a full behaviour planner C++ implementation. It will decide if the car should keep being in the same lane (KL), prepare for a lane change to the left or to the right (PLCL/PLCR) or perform a lane change to the left or to the right (LCL/LCR). In particular, the implemented parts of this challenge were **implementing the *choose_next_state()* function** in the *vehicle.cpp* and **choose the weights** in *cost.cpp*. This problem is aswell offered in Python for extra practice. [Here](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/November/5a024c18_python-3-practice/python-3-practice.zip) is the template and [here](http://video.udacity-data.com.s3.amazonaws.com/topher/2017/November/5a024c70_python-3-solution/python-3-solution.zip) the solution for the Python 3 version. Both the Python template and solution are in the *./02_behaviour_planner/python_practice/* folder.

NOTE: FSMs are a nice approach for solving Behaviour Planning when we do not have many states, such as in highway scenarios. For more complex situations, such as urban driving, other approaches might be more suitable.
