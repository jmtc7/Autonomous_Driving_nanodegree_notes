# Search Algorithms in Path Planning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Path planning determines where to go and how to do it keeping in mind the gathered data using both computer vision and sensor fusion. We will also use localization to know where we are in our environment.

We will start by learning **search algorithms** used in path planning. The next step will be to study how to **predict** where the objects around us will be in the future. Next, we will learn about **behaviour planning**, which is decide at high level what we should do. We will finish this module studying **trajectory generation** to create smooth, drivable and colission-free trajectories for the car.

The final project of the module will be a path planning algorithm for highways that will allow a vehicle to keep safe distances to other object, avoid collisions and even pass other vehicles if they go too slow.


## Basic Grid Search
Usually, a planning problem uses a starting and a goal position, a map and a cost function in order to reach the goal position starting from the starting one with the minimum possible cost. The cost is an important fact because if for whatever reason a certain movement is very expensive (maybe it is risky or implies a lot of energy), we may want to perform a phisically longer path whose cost will be lower. The iPython Notebook *00_expansion_search.ipynb* implements this by trying to advance in the given directions from each cell of the map, starting from the starting position and checking if it has reached the goal.

We can keep track of how many expansions we had to do to reach our solution by creating what is known as a *expansion grid*. This is a map that will contain *-1* if a cell has not been visited and the expansions that have been done in order to reach it in other case. This is implemented in *01_expansion_grid.ipynb*.

The final version of this will output the map with a path inside of it saying which movement we need to do in each cell to get from the starting position to the goal one. This one is called *02_draw_path.py*.


## A-star algorithm
It is a more efficient version of the basic search (which expands every node of the grid). It avoid expanding nodes in directions that are not efficient using a ***heuristic function***, which assigns certain values to each cell, which are the amount of steps necessary to get to the goal if there was no obstacle. This is done by using an **optimistic guess** of how far we are from the goal. We will use this heuristic function to decide which direction to expand in case of ties in the cost.

The heuristic function will vary depending on our available movements. If we can move in any directions, we may want to use the Euclidean Distance, whereas if we are limited to 4 movements (right, left, up and down), the Manhattan Distance may be more convinient.

We use the heuristic function adding each cell's *heuristic value* to the accumulated cost when we arrive to them and using this value to decide towards which cell to expand. As can be observed in *03_a_star.py*, the implementation is very straightforward given what we already coded.

NOTE: In the course, a video showing that A-star was used for dynamic planning (in an unmapped maze or finding obstacles in a map). However, some people pointed out that this algorithm is not meant for this purpose, suggesting the usage of [Rapidly-exploring Random Trees (RRT)](https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree) for this kind of undetermined applications.


## Dynamic Programing 
Its main characteristic is that its output are the best paths from every cell to a goal position, not only from a given starting point. This can be useful to generate alternative plans if something unexpected happens that will not allow us to follow our planned path.

This way, Dynamic Programming will provide a map in which all non-obstacle cells have a label, known as *policy*, which tells us which is the optimal action to do in order to reach a given goal position.

The policy is calculated by using a ***value function***, which is similar to the heuristic function used in A-star. This will provide a value to each cell telling the cost of arriving from there to the goal. In order to assign these values, we will start from the goal position, which will have a value of 0. All its neighbours will have a value of 1, whose neigbours will have a value of 2 and so on. Each cell will have a value equal to its neighbour with the smallest value + 1.

This algorithm is implemented in the *04_dynamic_programming.py* script. The policy map is computed in the *05_policy_map.py* script.


## Car-like Application
The last challenge of this lesson is with a car that has a 3D state consisting on its X and Y positions and its orientation, that will only be able to be up, down, right or left. The car will be able to perform 3 movements: (1) right turn, (2)no turn  and (3) left turn. The right turns cost 2 while going straight and turning left, 20. This is implemented in the script *06_policy_car.py*. The major difference with the other scripts is that now the solution space has 3 dimensions and it has to be reduced to a 2D projection for visualization purposes.




