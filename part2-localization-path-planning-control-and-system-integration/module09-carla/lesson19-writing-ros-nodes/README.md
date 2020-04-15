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


