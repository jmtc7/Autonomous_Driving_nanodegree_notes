# Introduction to ROS

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The Robot Operating System (ROS) is an open source robotics framework that provides libraries and tools for working with hardware and for comunicating different modules of a bigger system. The Udacity's autonomous vehicle (Carla) uses ROS. It also includes tools for visualization, simulation and analysis.

## ROS History
Its development started in the mid 2000s, as a project from the Standford's Artificial Intelligence Laboratory. In 2007, ROS became a formal entity supported by the company *Willow Garage*. In 2013, the Open Source Robotics Foundation has been the one developing and maintainig ROS. What motivated its development was the fact that many research groups were doing the same work as everybody else, losing time and making it harder to share advances and compare results.

## ROS Ecosystem
### Nodes and Topics
Given that many robots share the same architecture (uses perception for making decisions on how to actuate), this was the high-level model that the ROS developers have in mind. The individual tasks of these three high level modules are implemented in small software components called ***nodes***. They are very specific pieces of software responsible of a single task in the overall robot system, such as the control of the wheels, the localization or getting the data of one of the sensors.

All these nodes are coordinated by the ***ROS Master***, which keeps track of all the active nodes and makes it possible for them to reach and comunicate with each other. It also hosts the ***parameter server***, which stores parameters and settigns shared among all the nodes (e.g. the radius of the robot wheels, which could be useful for several nodes). This prevents us from storing the same information in several places and forget updating it somewhere.

### Messages and Services
The nodes communicate between them by sending ***messages*** using ***topics***. When a node sends messages, it becomes a ***publisher*** of the topic it is using to do so. When another one reads them, it will be a ***subscriber***. A single node can be both publisher and subscriber of several topics.

Each ROS distribution provides a set of predefined message types oriented to communicate physical variables (positions, velocities, accelerations, durations, rotations, etc.) or sensor readings (laser scans, images, pointclouds, etc.). Even a full ROS installation includes more than 200 message types, the ROS users are able to create new types combining the existing ones.

Even nodes will be able to send and receive messages to communicate between each other, it is useful to be able to have some sort of *request-respond* communication. This is where ***services*** come into play. ROS services allows communication between nodes, such as topics, but they are not *buses* that can be published or subscribed by many nodes. Services are 1-to-1 communications for things the developer want to happen once. An example would be a node wanting a new image. Even the node in charge of the interaction with the camera may be publishing every received image in a topic, it could be convenient to define a service so that a node will require an image specifying, for example, the exposure time or other parameters, and the node responsible for the camera will answer with a single image taking into account the requested things.

### Compute Graphs
They are graphical representations of the active nodes and in which topics they are publishing/subscribing. They also include the message type of each topic. They can be built by using ***rqt_graph***.


## Basic ROS commands
In order to run a basic ROS configuration, we could run these next commands in 3 different command line sessions:

```
roscore                            # Launches the ROS Master
rosrun turtlesim turtlesim_node    # Run the turtle simulation
rosrun turtlexim turtle_teleop_key # Run the node to teleoperate the turtle using the arrow keys
```

Now it is possible to use the arrow keys (from the command line session in which this node was launched) to teleoperate the turtle. i.e. move and turn the turtle. As we move the turtle, it will draw its trajectory.

Some other relevant commands are:

- `rosnode list`: Lists the active nodes.
- `rostopic list`: Lists the active topics.
- `rostopic info /topic`: Gives more information about the topic "/topic", such as the types of messages that can be sent through it and which nodes are publishing on it and which are subscribed to it.
- `rosmsg info msg_pkg/msg_type`: Prints information about the *msg_type* message type, which is contained in the package *msg_pkg*.
- `rosed msp_pkg msg_type.msg`: Edits the *msg_type.msg* file from the package *msg_pkg*, which contains the message definition. This can be used to modify the message type or just to get more information about it, such as the comments the developer included in this definition.
- `rostopic echo /topic`: Prints what is being published in the topic */topic*.

