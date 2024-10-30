#!/usr/bin/env python3

import os
import random
import copy
import threading
import time

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
        
        # Initialize YOLO model for object detection
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        path = os.path.abspath(__file__)
        src_dir = os.path.dirname(path)
        parent_dir = os.path.abspath(os.path.join(src_dir, '..', '..'))
        model_path = os.path.join(parent_dir, 'cam_assist/src/test_train/yolov11s_trained_optimized.pt')
        self.model_ = YOLO(model_path)
        rospy.loginfo(f"Using YOLO model on device: {self.device_}")

        # Depth and scan data handling
        # self.depth_data_ = None
        # self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ SE3(0.005, 0.028, 0.013)
        # self.scan_lock = threading.Lock()
        # self.scan_data = None
        
        self.current_map_ = None
        self.mineral_artefacts = []
        self.mushroom_artefacts = []       

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.WAITING_FOR_MAP

        self.goal_counter_ = 0  # Unique ID for each goal sent to move_base
        self.exploration_done_ = False
        self.visited_frontiers = set()

        # Initialise CvBridge for image processing
        self.cv_bridge_ = CvBridge()

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
        rospy.sleep(0.5)
    ####################################################################################################################
    
    
    def main_loop(self):

        while not rospy.is_shutdown():

            # Get the current status - possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number: ' + str(action_state))
            
            if (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully explored!")
                self.exploration_done_ = True
            elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE):
                print("Exploration preempted!")
                self.exploration_done_ = False
                action_state = actionlib.GoalStatus.PREEMPTING

            # Select the next planner to execute - Update this logic as you see fit!
            # if not self.exploration_done_:
            #     self.planner_type_ = PlannerType.WAITING_FOR_MAP

            # Execute the planner by calling the relevant method - methods send a goal to "move_base" with "self.move_base_action_client_"
            print("Calling planner:", self.planner_type_.name)
            self.exploration_planner(action_state)

            # Delay so the loop doesn't run too fast
            rospy.sleep(0.5)


    def exploration_planner(self, action_state):
        if action_state != actionlib.GoalStatus.ACTIVE:
            print('Exploration planner ............')
            if self.planner_type_ == PlannerType.WAITING_FOR_MAP:
                print ( 'enteringg .... waiting for map')
                self.handle_waiting_for_map()

            elif self.planner_type_ == PlannerType.IDENTIFYING_FRONTIERS:
                print ( 'entering ....IDENTifying frontiers')
                self.handle_identifying_frontiers()

            elif self.planner_type_ == PlannerType.SELECTING_FRONTIER:
                print ( 'entering ....SELECTING frontiers')
                self.handle_selecting_frontier()

            elif self.planner_type_ == PlannerType.MOVING_TO_FRONTIER:
                print ( 'entering ....MOVING TO frontiers')
                self.handle_moving_to_frontier(action_state)

            elif self.planner_type_ == PlannerType.HANDLE_REJECTED_FRONTIER:
                print ( 'entering ....HANDLE REJECTED frontiers')
                self.handle_selecting_frontier()
                
            elif self.planner_type_ == PlannerType.HANDLE_TIMEOUT:
                print ( 'entering ....HANDLE TIMEOUT')
                self.handle_timeout()

            elif self.planner_type_ == PlannerType.EXPLORED_MAP:
                rospy.loginfo("Exploration completed successfully.")
                
            # elif self.planner_type_ == PlannerType.OBJECT_IDENTIFIED_SCAN:
            #     rospy.loginfo("Object identified.")
            #     self.object_identified_scan()
                            

    def handle_waiting_for_map(self):
        print ('waiting for map.....')
        while self.current_map_ is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(1.0)  # Wait for the map to be received
        print ('map acquired')  
        
        self.planner_type_ = PlannerType.IDENTIFYING_FRONTIERS
        print ('state changed to identifying frontiers')
    

    def handle_identifying_frontiers(self):
        print ( 'identifying frontiers,,,,,')
        frontiers = self.identify_frontiers(self.current_map_)
        if not frontiers:
            rospy.loginfo('No frontiers found!')
            self.planner_type_ = PlannerType.EXPLORED_MAP
        else:
            self.planner_type_ = PlannerType.SELECTING_FRONTIER
            

    def handle_selecting_frontier(self):
        print ( ' frontiers found selecting frontiers')
        self.selected_frontier = None
        frontiers = self.identify_frontiers(self.current_map_)
        rospy.loginfo(f'Frontiers: {len(frontiers)}')
        rospy.loginfo(f'Frontiers after min dist select: {len(frontiers)}')
        self.selected_frontier = self.select_nearest_frontier(frontiers)
        if self.selected_frontier is None:
            rospy.logwarn('No frontier selected.')
            self.planner_type_ = PlannerType.EXPLORED_MAP
        else:
            
            rospy.loginfo('Frontier selected')
            self.planner_type_ = PlannerType.MOVING_TO_FRONTIER


    def handle_moving_to_frontier(self, action_state):
        
        # As robot moves to the frontier, it will check object detection - if object detected, it will stop and move towards the object
        # Implement logic for changing planner to object detection
        
        # Ensure there is a selected frontier to attempt
        while self.selected_frontier:
            # Take the current frontier to move towards
            frontier = self.selected_frontier[0]

            # Check if the current frontier is already visited
            if tuple(frontier) in self.visited_frontiers:
                # Remove the visited frontier and continue with the next one
                self.selected_frontier.pop(0)
                continue

            # Attempt to move to the selected frontier
            self.move_to_frontier(frontier)
            rospy.loginfo('Moving to frontier')
            start_time = rospy.Time.now()

            # Add this frontier to visited and remove it from the selected list
            removed_frontier = self.selected_frontier.pop(0)
            self.visited_frontiers.add(tuple(removed_frontier))

            # Continuously check the state while moving toward the goal
            while rospy.Time.now() - start_time < rospy.Duration(15):  # 10-second timeout
                action_state = self.move_base_action_client_.get_state()

                # Check if the goal has been reached successfully
                if action_state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Successfully reached the frontier!")
                    self.planner_type_ = PlannerType.WAITING_FOR_MAP
                    return  # Stop processing further frontiers in this call

                # Check if the goal was rejected or aborted
                elif action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                    rospy.loginfo("Goal rejected or aborted.")
                    self.planner_type_ = PlannerType.HANDLE_REJECTED_FRONTIER
                    break  # Break out of the inner loop to attempt the next frontier
                
            # If the loop ends due to timeout, handle the timeout case
            if rospy.Time.now() - start_time >= rospy.Duration(15):
                rospy.loginfo("Timeout reached.")
                self.planner_type_ = PlannerType.WAITING_FOR_MAP
                break  # Stop processing further frontiers in this call

        # Check if no more frontiers are available
        if not self.selected_frontier:
            rospy.loginfo("No more frontiers to explore.")
            self.planner_type_ = PlannerType.EXPLORED_MAP
            
            
    def handle_timeout(self):
        rospy.loginfo("Timeout reached.")
        self.current_map_ = None
        self.selected_frontier = None
        self.planner_type_ = PlannerType.WAITING_FOR_MAP
        
                                        
    def identify_frontiers(self, current_map): 
        frontiers = []
       # Extract map dimensions and data
        width =current_map.info.width
        height =current_map.info.height
        
        map_array = np.array(current_map.data).reshape((height, width))

        for i in range(height):
            for j in range(width):
                if map_array[i, j] == 0:  # Free space
                    neighbors = self.get_neighbors(i, j, map_array)
                    for n in neighbors:
                        if map_array[n[0], n[1]] == -1:  # Unknown space
                            frontiers.append((i, j))
                            break
        
        return frontiers


    def get_neighbors(self, row, col, map_array):
        neighbors = []
        height, width = map_array.shape
        if row > 0: neighbors.append((row-1, col))
        if row < height-1: neighbors.append((row+1, col))
        if col > 0: neighbors.append((row, col-1))
        if col < width-1: neighbors.append((row, col+1))
        return neighbors
  
    
    def select_nearest_frontier(self, frontiers):
        return self.merge_close_frontiers_to_one(frontiers)
    
    
    def merge_close_frontiers_to_one(self, frontiers):
        merged_frontiers = []
        threshold_distance = 50  # Distance threshold to consider frontiers as close

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

            # Merge the close frontiers into one
            if close_frontiers:
                avg_x = sum(f[0] for f in close_frontiers) / len(close_frontiers)
                avg_y = sum(f[1] for f in close_frontiers) / len(close_frontiers)
                merged_frontiers.append([avg_x, avg_y])

        return merged_frontiers
        
        
    def move_to_frontier(self, frontier):
        map_resolution = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position

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



if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('cave_explorer')
    # Create the cave explorer
    cave_explorer = CaveExplorer()
    # Loop forever while processing callbacks
    cave_explorer.main_loop()