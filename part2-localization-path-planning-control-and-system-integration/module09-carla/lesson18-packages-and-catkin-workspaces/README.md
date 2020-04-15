# Packages and Catkin Workspaces

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Catkin
It is the package management system provided by ROS. A Catkin workspace is a directory to build, modify and install packages. Usually, a single Catkin workspace will be used for a whole project. It will contain all the project packages, which are directories containing resources such as the source code of the nodes (contained in the *src* directory if they are in C++ or in the *scripts* directory if they are in Python), message definitions (in the *msg* directory, with a *.msg* extension), launchfiles (in the *launch* directory, with a *.launch* extension), service definitions (in the *srv* directory, with *.srv* extension), etc.

### Create a Workspace
To create a Catkin workspace named *catkin_ws* in our user's home directory, the next commands should be executed:

```
mkdir -p ~/catkin_ws/src # Create a directory for the workspace and a "src" directory inside of it
cd ~/catkin_ws/src       # Move to the created "src" directory
catkin_init_workspace    # Initialize the catkin workspace
```

After doing this, inside the *catkin_ws* directory, there will be a fully configured Catkin workspace with all the necessary files and folders to start developing ROS packages inside the *src* directory and compile them with the Catkin tools. This compilation is done by executing `catkin_make` in the root of the workspace (*~/catkin_ws/*), which will compile the uncompiled source code of all the packages in the *src*.

### Adding an Existing Package
Once having the workspace setted up, it is possible to add packages to it by cloning the repositories where the package of interest is contained in the *src* directory. This is done by:

```
cd ~/catkin_ws/src               # Move to the "src" directory inside the Catkin workspace
git clone [link_to_package_repo] # Clone package's repository in the current directory
cd ~/catkin_ws                   # Move to the root directory of the workspace
catkin_make                      # Compile all the projects in the "src" directory
```

It is also possible to just manually copy the package inside the *src* folder, as well as to install packages using *apt install* (or whichever package manager you use) or *rosdep install*.

### Creating New ROS Packages
To create a package, the **`catkin_create_pkg` command** is used. It is necessary to add the name of the package that is being created and its dependencies as follows:

```
catkin_create_pkg <your_package_name> [dependency1 dependency2 …]
```

ROS packages use a common **file structure**, as briefly commented in the past lesson. The most common folders that are found inside a ROS package are:

- scripts: python executables.
- src: C++ source files.
- msg: for custom message definitions.
- srv: for service message definitions.
- include: headers/libraries that are needed as dependencies.
- config: configuration files.
- launch: provide a more automated way of starting nodes.

Some other less common ones used for **simulations** are:

- urdf: Universal Robot Description Files.
- meshes: CAD files in .dae (Collada) or .stl (STereoLithography) format.
- worlds: XML like files that are used for Gazebo simulation environments.



## Other Useful ROS Tools
For the next steps, the *simple_arm* package will be used. It can be downloaded by doing one of the following commands (assuming a ROS Melodic installation):

```
sudo apt install ros-melodic-controller-manager
rosdep install simple_arm
```

### Roslaunch
This is a ROS tool that allows ROS users to use their *launch files*, which are files that contain instructions for running nodes, launching other launch files, etc. It is relevant to mention that if no ROS Master is running, when using the *roslaunch* command, one will be started.

### Rosdep
This ROS tool searches for missing dependencies of a given package and installs them. Assuming a package called *simple_arm*, it is possible to use rosdep as follows:

```
rosdep check simple_arm      # Check missing dependencies of the package
rosdep install -i simple_arm # Installs the missing dependencies of the package
```

