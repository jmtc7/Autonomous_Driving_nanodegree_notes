# Writing ROS Nodes

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

In this lesson, the target is to create a nodes in Python that publish and subscribe to topics and a ROS service. The first node will be *simple_mover*. It will publish joint angle commands for the *simple_arm* simulation. The next node will be *arm_mover*, which will provide the *safe_move* service, which will allow the arm to move to any position in its workspace considered safe. The *safe zone* will be defined by minimum and maximum joint angles that will be defined by the ROS parameter server. Finally, the last node to be implemented will be the *look_away* node, which will subscribe to a topic where camera data is being published. When the camera looks to the sky (will detect that because the color will be uniform), the robot will be moved to a new position using the *safe_move* service.

All this will be done inside the [Udacity's *simple_arm* package](https://github.com/udacity/simple_arm), which was cloned into the *./catkin_ws/src/* folder of this repository. All the details are documented using comments in the files.


## Publishing With Python Nodes
In Python, ROS publishers typically have the following definition format, although other parameters and arguments are possible:

```
pub1 = rospy.Publisher("/topic_name", message_type, queue_size=size)
```

The `/topic_name` indicates which topic the publisher will be publishing to. The `message_type` is the type of message being published on `/topic_name`. 

ROS publishing can be either synchronous or asynchronous:

- **Synchronous publishing** means that a publisher will attempt to publish to a topic but may be blocked if that topic is being published to by a different publisher. In this situation, the second publisher is blocked until the first publisher has serialized all messages to a buffer and the buffer has written the messages to each of the topic's subscribers. This is the default behavior of a rospy.Publisher if the `queue_size` parameter is not used or set to None.
- **Asynchronous publishing** means that a publisher can store messages in a queue until the messages can be sent. If the number of messages published exceeds the size of the queue, the oldest messages are dropped. The queue size can be set using the `queue_size` parameter.

Once the publisher has been created as above, a message with the specified data type can be published as follows:

```
pub1.publish(message)
```

NOTE: In order for the system to be able to execute Python scripts, the desired interpreter to be used must be explicitly indicated in the scripts. This is done by adding `#!/usr/bin/env python` as the first line of the script file. It is also necessary to give execution permisons to the file using `chmod u+x file_name.py`.


## Using ROS Services
A ROS service allows request/response communication to exist between nodes. Within the node providing the service, request messages are handled by functions or methods. Once the requests have been handled successfully, the node providing the service sends a message back to the requester node. In Python, a ROS service can be created using the following definition format:

```
service = rospy.Service('service_name', serviceClassName, handler)
```

Here, the `service_name` is the name given to the service. Other nodes will use this name to specify which service they are sending requests to.

The `serviceClassName` comes from the file name where the service definition exists. Each service has a definition provided in an *.srv* file; this is a text file that provides the proper message type for both requests and responses.

The `handler` is the name of the function or method that handles the incoming service message. This function is called each time the service is called, and the message from the service call is passed to the `handler` as an argument. The `handler` should return an appropriate service response message.



On the other hand, to use a ROS service from within another node, defininig a `ServiceProxy` is necessary, which provides the interface for sending messages to the service:

```
service_proxy = rospy.ServiceProxy('service_name', serviceClassName)
```

One way the `ServiceProxy` can then be used to send requests is as follows:

```
msg = serviceClassNameRequest()
#update msg attributes here to have correct data
response = service_proxy(msg)
```

In the code above, a new service message is created by calling the `serviceClassNameRequest()` method. This method is provided by rospy, and its name is given by appending `Request()` to the name used for `serviceClassName`. Since the message is new, the message attributes should be updated to have the appropriate data. Next, the `service_proxy` can be called with the message, and the response stored.


### Further Notes About ROS Services
It is possible to see a list of the currently available services by using `rosservice list`. Using, for example, `rosserviece list | grep safe`, only the ones containing *safe* on their names will be shown.

The robot's camera outpu can be visualize using `rqt_graph` as follows: `rqt_image_view /rgb_camera/image_raw`. The output shown by the last command will change whenever the robot moves, which can be done using the ROS service requesting new positions using the following command:

```
rosservice call /arm_mover/safe_move "joint_1: 1.57 joint_2: 1.57" # Example 1
rosservice call /arm_mover/safe_move "joint_1: 1.57 joint_2: 1.57" # Example 2
```

In order to modify parameters from the ROS Parameter Server, `rosparam set` is used:

```
rosparam set /arm_mover/max_joint_2_angle 1.57
```


## Subscribing With Python Nodes
A Subscriber enables a node to read messages from a topic, allowing useful data to be streamed into the node. In Python, ROS subscribers frequently have the following format, although other parameters and arguments are possible:

```
sub1 = rospy.Subscriber("/topic_name", message_type, callback_function)
```

The `/topic_name` indicates which topic the Subscriber should listen to. The `message_type` is the type of message being published on `/topic_name`. The `callback_function` is the name of the function that should be called with each incoming message. Each time a message is received, it is passed as an argument to `callback_function`. Typically, this function is defined in your node to perform a useful action with the incoming data. Note that unlike service handler functions, the `callback_function` is not required to return anything.


## ROS Logging Overview
In the codes of this lesson's nodes, logging statements such as the following ones can be found:

```
rospy.logwarn('j1 is out of bounds, valid range (%s,%s), clamping to: %s', min_j1, max_j1, clamped_j1) # Example 1
rospy.loginfo('GoToPositionRequest Received - j1:%s, j2:%s', req.joint_1, req.joint_2)                 # Example 2
```

Logging statements allow ROS nodes to send messages to a log file or the console. This allows errors and warnings to be surfaced to the user, or log data to be used later for debugging. By default all logging messages for a node are written to the node's log file which can be found in `~/.ros/log` or `ROS_ROOT/log` . If a ROS Master is running, it is possible to use `roscd` to find log file directory by opening a new terminal window and typing `roscd log`. The `latest` directory will contain the log files of the most recent run.

### Logging Levels
There are several levels of log messages offered by rospy. These are the following:

```
rospy.logdebug(...)
rospy.loginfo(...)
rospy.logwarn(...)
rospy.logerr(...)
rospy.logfatal(...)
```

Depeding the level assigned to a logged message, it will be written to one or another output, as explained [here](http://wiki.ros.org/rospy/Overview/Logging). Only the *info* will go to stdout, the *warnings*, *errors* and *fatals* will go to stderr, everything will go to the log file and only the *debugging* logs will be excluded from */rosout*.

It is possible to filter the levels by *grepping* the `rostopic echo` output of the `/rosout` node. This can be done as follows:

```
rostopic echo /rosout | grep insert_search_expression_here
```

Or like that if redirecting this output to some file is desired:

```
rostopic echo /rosout | grep insert_search_expression_here > path_to_output/output.txt
```


It is also possible to make the debugging messages of a certain node to be sent to `/rosout` using the `log_level` parameter of the `init_node()` function like this:

```
rospy.init_node('my_node', log_level=rospy.DEBUG) # Other labels are INFO, WARN, ERROR, and FATAL
```

In the launch files, it is possible to set the `output` attribute of a node tag to `"screeen"` or `"log"`. The first option will display the messages sent to both the `stdout` and `stderr` in the screen. However, the `"log"` option will send the `stdout` to the log file and the `stderr` to both the screen and the log file. An example on how to do this for the `look_away` node would be:

```
<!-- The look away node -->
<node name="look_away" type="look_away" pkg="simple_arm" output="screen"/>
```


