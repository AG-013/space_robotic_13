cmake_minimum_required(VERSION 3.1.3)

project(path_planner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find_package(OpenCV 3 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        roslib
        nav_msgs
        geometry_msgs
        )

catkin_package()


