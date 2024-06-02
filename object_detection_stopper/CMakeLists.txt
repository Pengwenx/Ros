cmake_minimum_required(VERSION 3.0.2)
project(object_detection_stopper)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  robot_vision
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/object_detection_stopper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)