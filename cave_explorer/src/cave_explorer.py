#!/usr/bin/env python3

import os
import random
import copy
import threading
from enum import Enum
import time

# Math Modules
from spatialmath import SE3
import math
import numpy as np

# # Machine Learning / OpenCV Modules
# import cv2  # OpenCV2
# import torch
# from ultralytics import YOLO
# from cv_bridge import CvBridge, CvBridgeError

# ROS Modules
import tf
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Twist, Pose2D, Pose, Point , Quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import tf
import actionlib
import rospy
import roslib
from collections import deque
import heapq



class PlannerType(Enum):
    ERROR = 0
    WAITING_FOR_MAP = 1
    IDENTIFYING_FRONTIERS = 2
    SELECTING_FRONTIER = 3
    MOVING_TO_FRONTIER = 4
    HANDLE_REACHED_FRONTIER = 5
    HANDLE_REJECTED_FRONTIER = 6
    HANDLE_TIMEOUT = 7
    OBJECT_IDENTIFIED_SCAN = 8
    EXPLORED_MAP = 9


class CaveExplorer:
    
    def __init__(self):

        self.current_map_ = None

        # Variables/Flags for planning
        self.goal_counter_ = 0  # Unique ID for each goal sent to move_base
        self.exploration_done_ = False
        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
        self.visited_frontiers = set()

        # Wait for the transform from map to base_link
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        # Subscribers
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        
        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")        
                
        
    # Callbacks ###################################################################################################
    def map_callback(self, map_msg):
        self.current_map_ = map_msg
        rospy.sleep(0.2)
    ####################################################################################################################
    
    
    def main_loop(self):

        while not rospy.is_shutdown():

            # Get the current status, possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            # rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            # rospy.loginfo('action_state number:' + str(action_state))
    
            # Execute the planner by calling the relevant method - methods send a goal to "move_base" with "self.move_base_action_client_", Add your own planners here!
            print("Current State:", self.exploration_state_.name)
            self.exploration_planner(self.exploration_state_)

            # Delay so the loop doesn't run too fast
            rospy.sleep(0.1)


################ CHANGE STATES FUNCTIONING HIGH LEVEL ############################################################################################################
    def exploration_planner(self, action_state):

        if action_state != actionlib.GoalStatus.ACTIVE:
            print('Exploration planner ............')

            if self.exploration_state_ == PlannerType.WAITING_FOR_MAP:
                self.handle_waiting_for_map()

            elif self.exploration_state_ == PlannerType.SELECTING_FRONTIER or self.exploration_state_ == PlannerType.HANDLE_REJECTED_FRONTIER:
                self.handle_selecting_frontier()

            elif self.exploration_state_ == PlannerType.MOVING_TO_FRONTIER:
                print('entering ....MOVING TO frontiers')
                self.handle_moving_to_frontier(action_state)

            elif self.exploration_state_ == PlannerType.HANDLE_TIMEOUT:
                print('entering ....HANDLE TIMEOUT')
                self.handle_timeout()

            elif self.exploration_state_ == PlannerType.EXPLORED_MAP:
                rospy.loginfo("Exploration completed successfully.")

            elif self.exploration_state_ == PlannerType.OBJECT_IDENTIFIED_SCAN:
                rospy.loginfo("Object identified.")
                self.object_identified_scan()
               
 ##############################################################################################################################################################################    
     
    def handle_waiting_for_map(self):
        while self.current_map_ is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(0.5)  # Wait for the map to be received
        self.exploration_state_ = PlannerType.SELECTING_FRONTIER
                
            
    def handle_selecting_frontier(self):
        
        frontiers , visited = self.identify_frontiers(self.current_map_)
        # print ( ' visited:__checks ::', visited)
        print ( ' visited set : ' , len(self.visited_frontiers))

        if not frontiers:
            rospy.logwarn('No frontiers found.')
            self.exploration_state_ = PlannerType.EXPLORED_MAP
            return
        
        # filter out frontiers too close
        print  (' frontiers:', len(frontiers) )
        filtered_frontier = [f for f in frontiers if self.is_valid_frntr(f)]

        # Calculate cost for each frontier and sort them
        # print ('filtered frontiers lenght:', len(filtered_frontier))
        # print ('filtered frontiers 1st set:', filtered_frontier[0][:4])
        # print ('filtered frontiers 2nd set:', filtered_frontier[1][:4])
        # print ('filtered frontiers 3rd set:', filtered_frontier[2][:4])

        frontier_costs = [(frontier, self.frontier_cost(frontier)) for frontier in filtered_frontier if frontier is not None and len(frontier) > 5]
        print('frontier costs:', len(frontier_costs))
        
        # filtering out the invalid frontiers and inf cost frontiers    
        
        valid_frontier_costs = [(f,c) for f,c in frontier_costs if np.isscalar(c) and not np.isnan(c) and not np.isinf(c)]
        
        print('valid frontier costs:', len(valid_frontier_costs))
        
        if not valid_frontier_costs:
            rospy.logwarn('No valid frontiers found.')
            self.exploration_state_ = PlannerType.EXPLORED_MAP
            return
        
        # Sort the frontiers based on cost
        print('valid frontier costs:', valid_frontier_costs[0][1])
        sorted_frontiers = heapq.nsmallest(len(frontier_costs), frontier_costs, key=lambda x: x[1])
        print('sorted frontiers:', len(sorted_frontiers))
        print('sorted frontiers:', sorted_frontiers[0])
        # Select the frontier with the lowest cost
        self.selected_frontier = [frontier for frontier, cost in sorted_frontiers]
        print ('selected frontier:', len(self.selected_frontier))
        if not self.selected_frontier:
            rospy.logwarn('No frontier selected.')
            self.exploration_state_ = PlannerType.EXPLORED_MAP
        else:
            rospy.loginfo('Frontier selected')
            self.exploration_state_ = PlannerType.MOVING_TO_FRONTIER

    # As robot moves to the frontier, it will check object detection - if object detected, it will stop and move towards the object, Implement logic for changing planner to object detection
    def handle_moving_to_frontier(self, action_state):        
        # Ensure there is a selected frontier to attempt
        while self.selected_frontier:
            # Take the current frontier to move towards
            frontier = self.selected_frontier[0]
            print('frontier:', frontier)
            
            centroud_frntr = np.mean(frontier, axis=0).astype(int)

            # Check if the current frontier is already visited
            if tuple(centroud_frntr) in self.visited_frontiers:
            #     # Remove the visited frontier and continue with the next one
                if self.selected_frontier:
                    self.selected_frontier.pop(0)
                continue
            
            # # Add this frontier to visited and remove it from the selected list
            # removed_frontier = self.selected_frontier.pop(0)
            # self.visited_frontiers.add(tuple(removed_frontier))

            # Attempt to move to the selected frontier
            self.move_to_frontier(centroud_frntr)
            rospy.loginfo('Moving to frontier')
            start_time = rospy.Time.now()
            
            self.visited_frontiers.add(tuple(centroud_frntr))
            
            if self.selected_frontier is not None:
                self.selected_frontier.pop(0)
                
            # Continuously check the state while moving toward the goal
            while rospy.Time.now() - start_time < rospy.Duration(8):  # 10-second timeout
                action_state = self.move_base_action_client_.get_state()

                # Check if the goal has been reached successfully
                if action_state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Successfully reached the frontier!")
                    self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                    return  # Stop processing further frontiers in this call

                # Check if the goal was rejected or aborted
                elif action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                    rospy.loginfo("Goal rejected or aborted.")
                    self.exploration_state_ = PlannerType.HANDLE_REJECTED_FRONTIER
                    break  # Break out of the inner loop to attempt the next frontier
                
            # If the loop ends due to timeout, handle the timeout case
            if rospy.Time.now() - start_time >= rospy.Duration(8):
                rospy.loginfo("Timeout reached.")
                self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                break  # Stop processing further frontiers in this call

        # Check if no more frontiers are available
        if not self.selected_frontier:
            rospy.loginfo("No more frontiers to explore.")
            self.exploration_state_ = PlannerType.EXPLORED_MAP
            
            
    def handle_timeout(self):
        rospy.loginfo("@ handle_timeout")
        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
        
       
       
############# FRONTIER BASED FUNCTIONS ##############################################################################################################       
    def identify_frontiers(self, current_map):
        frontiers = [] # empty so each loop they are re calculated
       # Extract map dimensions and data
        width = current_map.info.width
        height = current_map.info.height

        map_array = np.array(current_map.data).reshape((height, width))
        visited = np.zeros_like(map_array, dtype=bool) # same size array as map_array with bool flags

        for i in range(height):
            for j in range(width):
                    # checking if cell is visited and is frontier
                if self.is_frontier(map_array, i, j) and not visited[i, j]:
                    
                    # print ( ' frontier found at:', i, j)
                     # cheking to separate the frontiers
                    frontier = self.bfs_frontier(map_array, i, j, visited)
                    if frontier:
                        # check if it is in visited
                        # print ( ' new frontier set:', len(frontier))
                        frontier_cntr = tuple(np.mean(frontier, axis=0).astype(int))
                        if frontier_cntr not in self.visited_frontiers:
                            frontiers.append(frontier)
        return frontiers , visited


    def is_frontier(self, map_array , i , j):
        # Check if the cell is a free cell
        if map_array[i, j] != 0:
            return False
        
        # check if the cell has an unknown cell as neighbor
        neighbors = self.get_neighbors(i, j, map_array)
        
        for ni, nj in neighbors:
            if map_array[ni, nj] == -1:
                return True
        return False
                
    def bfs_frontier(self, map_array, i, j, visited):
        
        # if (i, j) in visited:
        #     return []
        
        queue = deque([(i, j)])   # frontier queue
        frontier = []             # frontier list
        
        while queue:
            i, j = queue.popleft()
            # cell = i, j
            if visited[i, j]:
                continue
                        
            visited[i, j] = True # mark as visited
            
            #check if the cell is frontier
            if self.is_frontier(map_array, i, j):
                frontier.append([i, j])
                
            # All unvisited neighbors of the queue cell
            
            for ni, nj in self.get_neighbors(i, j, map_array):
                if not visited[ni, nj] and map_array[ni, nj] == 0:
                    queue.append((ni, nj))
                    
        return frontier if frontier else None                       
                        
                        
    def get_neighbors(self, row, col, map_array):
        neighbors = []
        height, width = map_array.shape
        
        if row > 0:         neighbors.append((row-1, col))
        if row < height-1:  neighbors.append((row+1, col))
        if col > 0:         neighbors.append((row, col-1))
        if col < width-1:   neighbors.append((row, col+1))
        # print('neighbors:', neighbors)
        return neighbors

#########################
    def is_valid_frntr (self, frontier) :
        # min dist threshold for near frnts
        min_dist = 30
        for visited_point in self.visited_frontiers:
            for pt in frontier:
                dist = np.hypot((pt[0] - visited_point[0]) , (pt[1] - visited_point[1]))
                if dist < min_dist:
                    return False
                    #frontier too close 
        return True
                
    def frontier_cost(self, frontier):
        
        if frontier is None or len(frontier) == 0:
            rospy.logwarn('No frontier found.')
            return float('inf')
        
        distance = self.compute_distance_to_frntr(frontier)
        size = len(frontier)  # size of the frontier
        # orientation = self.compute_orn_frn (frontier)
        
        l_1 = 0.1
        l_2 = 0.8
        l_3 = 0.3
        
        cost = l_1 * distance + l_2 * size  #l_3 * orientation
        return cost

    def compute_distance_to_frntr(self, frontier):
        # Ensure that frontier contains multiple points
        if len(frontier) < 2:
            # rospy.logwarn("Frontier does not contain enough points, assuming single point frontier.")
            frontier_cntr = frontier[0]  # Take the single point directly
        else:
             # Validate that all elements in frontier are coordinate pairs
            valid_frontier = [pt for pt in frontier if len(pt) == 2]
            if not valid_frontier:
                rospy.logerr("Invalid frontier detected. No valid coordinate pairs found.")
                return float('inf')  # Return a high distance to ignore this invalid frontier
            # Calculate the centroid of the frontier
            frontier_cntr = np.mean(valid_frontier, axis=0).astype(int)
        
        map_res = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        
        current_position = self.get_pose_2d()
        
        # Calculate the real-world coordinates of the frontier centroid
        frontier_x = frontier_cntr[0] * map_res + map_origin.x
        frontier_y = frontier_cntr[1] * map_res + map_origin.y
        
        # Compute the Euclidean distance from the robot's current position to the frontier centroid
        distance = np.hypot((frontier_x - current_position.x), (frontier_y - current_position.y))
        
        return distance
         
    def compute_orn_frn(self, frontier):
        """
        Calculate the orientation cost of a given frontier.
        """
        # Get the map properties
        map_res = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position

        # Assuming frontier is a collection of cells (x, y)
        total_orientation = 0
        for cell in frontier:
            # Validate that cell is a coordinate pair (x, y)
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                rospy.logwarn(f"Unexpected cell format: {cell}. Skipping...")
                continue

            frontier_x = cell[0] * map_res + map_origin.x
            frontier_y = cell[1] * map_res + map_origin.y

            # Compute the angle/orientation of this frontier cell with respect to the robot's current position
            robot_pose = self.get_pose_2d()
            angle_to_cell = math.atan2(frontier_y - robot_pose.y, frontier_x - robot_pose.x)
            
            # Calculate the relative orientation cost (custom logic, modify as needed)
            orientation_diff = abs(angle_to_cell - robot_pose.theta)
            total_orientation += orientation_diff

        # Calculate an average orientation if there are multiple cells
        avg_orientation = total_orientation / len(frontier) if frontier else 0

        return avg_orientation
                
    def wrap_angle(self, angle):
        # Function to wrap an angle between 0 and 2*Pi
        while angle < 0.0:
            angle = angle + 2 * math.pi

        while angle > 2 * math.pi:
            angle = angle - 2 * math.pi

        return angle
 
    def pose2d_to_pose(self , pose_2d):
        pose = Pose()

        pose.position.x = pose_2d.x
        pose.position.y = pose_2d.y

        pose.orientation.w = math.cos(pose_2d.theta / 2.0)
        pose.orientation.z = math.sin(pose_2d.theta / 2.0)

        return pose

    def move_to_frontier(self, centroud_frntr):
        
        map_resolution = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        # send first selected frontier to move_base and remove it from, till all the frontiers are sent to move_base
        x, y = centroud_frntr
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
        action_goal.goal.target_pose.pose = self.pose2d_to_pose(pose_2d)

        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)

           
    def object_identified_scan(self):
        # Stop the robot
        # twist = Twist()
        # self.cmd_vel_pub_.publish(twist)
        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
    
    def get_pose_2d(self):
        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = trans[0]
        pose.y = trans[1]
        qw = rot[3]
        qz = rot[2]

        if qz >= 0.:
            pose.theta = self.wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = self.wrap_angle(-2. * math.acos(qw))

        return pose
       
        
if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('cave_explorer')
    # Create the cave explorer
    cave_explorer = CaveExplorer()
    # Loop forever while processing callbacks
    cave_explorer.main_loop()
    
