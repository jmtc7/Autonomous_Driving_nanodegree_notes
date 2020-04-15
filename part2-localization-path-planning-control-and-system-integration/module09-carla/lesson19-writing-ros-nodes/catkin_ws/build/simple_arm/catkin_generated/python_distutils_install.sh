#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/src/simple_arm"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/install/lib/python2.7/dist-packages:/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/build" \
    "/usr/bin/python2" \
    "/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/src/simple_arm/setup.py" \
    build --build-base "/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/build/simple_arm" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/install" --install-scripts="/home/jmtc7/Learning/self-driving-car-nanodegree/part2-localization-path-planning-control-and-system-integration/module09-carla/lesson19-writing-ros-nodes/catkin_ws/install/bin"
