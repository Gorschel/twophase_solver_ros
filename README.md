# Overviw
ROS-service of the famous two phase algorithm by Herbert Kociemba.
This project's base is forked from a early version from https://github.com/hkociemba/RubiksCube-TwophaseSolver.
I reordered the file locations and adjusted the code to run on python 2.7
My image processing approach for scanning a cube can be found inside `scripts/image_processing.py`
Note, that this approach only works for black cube bases and homogenous matte sticker colors.

# Usage
1. build your catkin workspace
2. `$ roslaunch twophase_solver_ros solver.launch`
3. have another ROS node sending a "request" in form of a CubeDefinitionString (example can be found at `scripts/cubedefstring.example`)


# ROS
may work for newer versions of ROS, but was designed to run on melodic