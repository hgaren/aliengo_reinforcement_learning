cmake_minimum_required(VERSION 2.8.3)
project(learning_ros)

find_package(catkin REQUIRED COMPONENTS
  
  rospy
  roslaunch
)
catkin_package(
)

 

install(DIRECTORY config 
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
