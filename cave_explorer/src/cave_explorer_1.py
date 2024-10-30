#!/usr/bin/env python3

import os
import random
import copy
import threading
import time
import random

# Math Modules
from spatialmath import SE3
import math
import numpy as np

# Machine Learning / OpenCV Modules
import cv2 # OpenCV2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError

# ROS Modules
import tf
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid, Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Pose2D, Pose, Point
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from sensor_msgs import point_cloud2
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
from visualization_msgs.msg import Marker
import rospy
import roslib

from helper_functions import *
from enums import *


class CaveExplorer:
    
    def __init__(self):

        self.current_map = None

        self.planner_type_ = PlannerType.WAITING_FOR_MAP
        self.goal_counter_ = 0
        self.exploration_done_ = False

        self.cv_bridge_ = CvBridge()

        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")
        
 
    # Callbacks ###################################################################################################
    def map_callback(self, map_msg):
        self.current_map = map_msg
        rospy.sleep(0.1)
    ####################################################################################################################
    
    
    def main_loop(self):

        while not rospy.is_shutdown():
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('Action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('Action state number: ' + str(action_state))

            if self.planner_type_ == PlannerType.EXPLORATION and action_state == actionlib.GoalStatus.SUCCEEDED:
                print("Successfully explored!")
                self.exploration_done_ = True
            elif self.planner_type_ == PlannerType.EXPLORATION and action_state == actionlib.GoalStatus.ACTIVE:
                print("Exploration preempted!")
                self.exploration_done_ = False

            print("Calling planner:", self.planner_type_.name)
            self.exploration_planner(action_state)

            rospy.sleep(0.5)


    def exploration_planner(self, action_state):
        if action_state != actionlib.GoalStatus.ACTIVE:
            print('Exploration planner...')

            if self.planner_type_ == PlannerType.WAITING_FOR_MAP:
                self.handle_waiting_for_map()
            elif self.planner_type_ == PlannerType.SELECTING_FRONTIER or self.planner_type_ == PlannerType.HANDLE_REJECTED_FRONTIER:
                self.handle_selecting_frontier()
            elif self.planner_type_ == PlannerType.MOVING_TO_FRONTIER:
                self.handle_moving_to_frontier(action_state)
            elif self.planner_type_ == PlannerType.HANDLE_TIMEOUT:
                self.handle_timeout()
            elif self.planner_type_ == PlannerType.EXPLORED_MAP:
                rospy.loginfo("Exploration completed successfully.")
                rospy.sleep(0.1)
                            

    def handle_waiting_for_map(self):
        print('Waiting for map...')
        while self.current_map is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(1.0)
        print('Map acquired')
        self.planner_type_ = PlannerType.SELECTING_FRONTIER
            

    def handle_selecting_frontier(self):
        print('Selecting frontier...')
        frontiers = self.identify_frontiers()
        # rospy.loginfo(f'Frontiers: {len(frontiers)}') # Don't need to print this

        self.selected_frontier = self.select_nearest_frontier(frontiers, self.get_pose_2d_simple())
        if self.selected_frontier is None:
            rospy.logwarn('No frontier selected - NO FRONTIER TO EXPLORE')
            self.planner_type_ = PlannerType.EXPLORED_MAP
        else:
            self.planner_type_ = PlannerType.MOVING_TO_FRONTIER


    def identify_frontiers(self): 
        frontiers = []
        # Extract map dimensions and data
        width = self.current_map.info.width
        height = self.current_map.info.height

        # Reshape map data into a 2D array
        map_array = np.array(self.current_map.data).reshape((height, width))

        for i in range(height):
            for j in range(width):
                if map_array[i, j] == 0:  # Free space
                    neighbors = self.get_neighbors(i, j, map_array)
                    for n in neighbors:
                        if map_array[n[0], n[1]] == -1:  # Unknown space
                            frontiers.append((i, j))
                            break

        return frontiers


    def select_nearest_frontier(self, frontiers, current_position):
        # MERGING FRONTIERS ##################################################################
        merged_frontiers = []
        threshold_distance = 10  # Distance threshold to consider frontiers as close

        while frontiers:
            # Select a random frontier
            random_frontier = random.choice(frontiers)
            frontiers.remove(random_frontier)

            # Find close frontiers to the selected random frontier
            close_frontiers = [random_frontier]
            for frontier in frontiers[:]:
                if compute_distance_between_points(random_frontier, frontier) < threshold_distance:
                    close_frontiers.append(frontier)
                    frontiers.remove(frontier)

            # Merge the close frontiers into one average point
            if close_frontiers:
                avg_x = sum(f[0] for f in close_frontiers) / len(close_frontiers)
                avg_y = sum(f[1] for f in close_frontiers) / len(close_frontiers)
                # Ensure the merged frontier is within map boundaries
                avg_x = min(max(0, avg_x), self.current_map.info.height - 1)
                avg_y = min(max(0, avg_y), self.current_map.info.width - 1)
                merged_frontiers.append([avg_x, avg_y])
        ##########################################################################################

        # RETURN CLOSEST FRONTIER ################################################################
        if not merged_frontiers:
            return None  # No valid frontiers found

        # Compute distances from current position to each merged frontier and select the closest
        nearest_frontier = min(merged_frontiers, key=lambda f: compute_distance_between_points(current_position, f))
        
        return nearest_frontier
    

    def handle_moving_to_frontier(self, action_state):
        
        self.move_to_frontier(self.selected_frontier)
        rospy.loginfo('Moving to frontier')
        start_time = rospy.Time.now()
        
        max_time = 10
        
        while rospy.Time.now() - start_time < rospy.Duration(max_time):
            action_state = self.move_base_action_client_.get_state()
            if action_state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Successfully reached the frontier!")
                self.planner_type_ = PlannerType.WAITING_FOR_MAP
                return
            elif action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                rospy.loginfo("Goal rejected or aborted.")
                self.planner_type_ = PlannerType.HANDLE_REJECTED_FRONTIER
                break

        if rospy.Time.now() - start_time >= rospy.Duration(max_time):
            rospy.loginfo("Timeout reached.")
            self.planner_type_ = PlannerType.HANDLE_TIMEOUT
            
    
    def move_to_frontier(self, frontier):
        map_resolution = self.current_map.info.resolution
        map_origin = self.current_map.info.origin.position

        # send first selected frontier to move_base and remove it from till all the frontiers are sent to move_base
        x, y = frontier

        # Send a goal to move_base to explore the selected frontier
        pose_2d = Pose2D()

        # Move forward 10m
        pose_2d.x = x * map_resolution + map_origin.x
        pose_2d.y = y * map_resolution + map_origin.y
        pose_2d.theta = math.pi/2
        print(f'x:{pose_2d.x} , y:{pose_2d.y}')

        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ = self.goal_counter_ + 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)
        
                            
    def get_neighbors(self, row, col, map_array):
        neighbors = []
        height, width = map_array.shape
        if row > 0: neighbors.append((row-1, col))
        if row < height-1: neighbors.append((row+1, col))
        if col > 0: neighbors.append((row, col-1))
        if col < width-1: neighbors.append((row, col+1))
        return neighbors
    
    
    def handle_timeout(self):
        rospy.loginfo("Timeout reached. Cancelling goals and going back to waiting for map:")
        
        self.move_base_action_client_.cancel_all_goals()
        self.current_map = None
        self.selected_frontier = None
        self.planner_type_ = PlannerType.WAITING_FOR_MAP
        
        
    # X,Y only
    def get_pose_2d_simple(self):
        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
        return trans[0], trans[1]
    

if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('cave_explorer')
    # Create the cave explorer
    cave_explorer = CaveExplorer()
    # Loop forever while processing callbacks
    cave_explorer.main_loop()