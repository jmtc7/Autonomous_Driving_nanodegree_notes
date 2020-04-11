# Search Algorithms in Path Planning

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Path planning determines where to go and how to do it keeping in mind the gathered data using both computer vision and sensor fusion. We will also use localization to know where we are in our environment.

We will start by learning **search algorithms** used in path planning. The next step will be to study how to **predict** where the objects around us will be in the future. Next, we will learn about **behaviour planning**, which is decide at high level what we should do. We will finish this module studying **trajectory generation** to create smooth, drivable and colission-free trajectories for the car.

The final project of the module will be a path planning algorithm for highways that will allow a vehicle to keep safe distances to other object, avoid collisions and even pass other vehicles if they go too slow.

Usually, a planning problem uses a starting and a goal position, a map and a cost function in order to reach the goal position starting from the starting one with the minimum possible cost. The cost is an important fact because if for whatever reason a certain movement is very expensive (maybe it is risky or implies a lot of energy), we may want to perform a phisically longer path whose cost will be lower.

