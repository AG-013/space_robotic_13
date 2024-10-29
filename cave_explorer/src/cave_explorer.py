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

# Machine Learning / OpenCV Modules
import cv2  # OpenCV2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError

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


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    EXPLORATION = 6


class ExplorationsState(Enum):
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
        # Initialize YOLO model for object detection
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        path = os.path.abspath(__file__)
        src_dir = os.path.dirname(path)
        parent_dir = os.path.abspath(os.path.join(src_dir, '..', '..'))
        model_path = os.path.join(parent_dir, 'cam_assist/src/test_train/yolov11s_trained_optimized.pt')
        self.model_ = YOLO(model_path)
        rospy.loginfo(f"Using YOLO model on device: {self.device_}")

        # Depth and scan data handling
        self.depth_data_ = None
        self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ SE3(0.005, 0.028, 0.013)
        self.scan_lock = threading.Lock()
        self.map_lock = threading.Lock()
        self.image_lock = threading.Lock()
        self.odom_lock = threading.Lock()


        self.current_map_ = None
        self.image_data = None

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.goal_counter_ = 0
        self.exploration_done_ = False
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        self.visited_frontiers = set()

        # Initialize CvBridge for image processing
        self.cv_bridge_ = CvBridge()
        
        # Variables/Flags for perception
        self.localised_ = False
        self.artifact_found_ = False
        self.artefact_list = []


        # Wait for the transform from map to base_link
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        # Subscribers
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.depth_sub_ = rospy.Subscriber("/camera/depth/points", PointCloud2, self.depth_callback, queue_size=1)
        
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

        # Publishers
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_pub_ = rospy.Publisher("/detections_image", Image, queue_size=5)
        # self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Artifact markers

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Start the image processing thread
        self.image_thread = threading.Thread(target=self.process_image_continuously)
        self.image_thread.start()

        # Start the exploration thread
        self.exploration_thread = threading.Thread(target=self.exploration_loop)
        self.exploration_thread.start()



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

        print("pose: ", pose)

        return pose


    def get_posed_3d(self, pixel_x: int, pixel_y: int) -> tuple:
        # Check if depth data is available
        if not self.depth_data_:
            rospy.logwarn("Depth message not received yet!")
            return None

        # Get current robot pose transformation from 'map' to 'base_link'
        (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
        x, y, z = trans
        qz = rot[2]
        qw = rot[3]

        # Calculate theta based on quaternion values
        theta = self.wrap_angle(2.0 * math.acos(qw)) if qz >= 0.0 else self.wrap_angle(-2.0 * math.acos(qw))

        # Create robot pose transformation using SE3
        robot_pos = SE3(x, y, z) @ SE3.Rz(theta)  # Using Rz for 2D rotation instead of full RPY rotation

        # Extract point from depth data using pixel coordinates
        point = list(point_cloud2.read_points(
            self.depth_data_,
            field_names=("x", "y", "z"),
            skip_nans=True,
            uvs=[(pixel_x, pixel_y)]
        ))

        if point:
            # Convert point from camera optical frame to robot base frame
            old_x, old_y, old_z = point[0]
            
            # Transform from camera optical frame to camera frame (Realsense camera)
            x = old_z
            y = -old_x
            z = -old_y

            # Create point transformation
            point_transform = SE3.Trans(x, y, z)
            point_in_world = robot_pos @ self.base_2_depth_cam @ point_transform

            # Extract the translation components (t) from the final transformation
            return point_in_world.t  # .t extracts the translational component as a tuple (x, y, z)

        return None

    
    # Callback functions with locks #######################################################
    def image_callback(self, image_msg):
        with self.image_lock:
            self.image_data = image_msg
            
    def depth_callback(self, depth_msg):
        with self.scan_lock:
            self.depth_data_ = depth_msg

    def map_callback(self, map_msg):
        with self.map_lock:
            if self.current_map_ is None or self.current_map_.header.stamp != map_msg.header.stamp:
                rospy.loginfo("Map updated")
                self.current_map_ = map_msg
                
    # Image processing thread ############################################################
    def process_image_continuously(self):
        while not rospy.is_shutdown():
            with self.image_lock:
                if self.image_data:
                    self.process_image()
            rospy.sleep(0.1)

    def process_image(self):
        try:
            classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]

            cv_image = self.cv_bridge_.imgmsg_to_cv2(self.image_data, "bgr8")
            # Perform image detection or processing here (e.g., YOLO, feature detection, etc.)

            # Process the image using YOLO
            # print('------------------------------------------------------------')
            # print('USING CUDA:', self.device_)
            # print('------------------------------------------------------------')
            results = self.model_(cv_image, device=self.device_, imgsz=(480, 384))

            
            # 
            mineral_artefacts = []
            mushroom_artefacts = []
            
            # Draw bounding boxes on the image
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf[0].item()  # Confidence score

                    # Only process boxes with confidence above the threshold
                    confidence_threshold = 0.5
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_id = int(box.cls.item())  # Get the class ID from the tensor
                        label = f'{classes[class_id]} {confidence:.2f}'  # Class name and confidence

                        # Calculate and print center point of the bounding box
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        # rospy.loginfo(f"Center of {classes[class_id]}: ({center_x}, {center_y})")

                        # Get the 3D coordinates
                        self.art_xyz = self.get_posed_3d(center_x, center_y)

                        # Check if art_xyz is None before accessing its elements
                        if self.art_xyz is not None:
                            # Draw rectangle and label
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

                            # Detecting Mineral and Mushroom
                            if class_id == 1 or class_id == 4:
                                if class_id == 1:
                                    self.artefact_list = mineral_artefacts
                                else:
                                    self.artefact_list = mushroom_artefacts
                                    
                                    self.exploration_state_ = ExplorationsState.OBJECT_IDENTIFIED_SCAN
                                
                                # Check if it doesn't already exist
                                already_exists = False
                                _art_art_dist_threshold = 7  # Minimum distance threshold between artefacts
                                for artefact in self.artefact_list:
                                    if self.is_artifact_far_enough(artefact, self.art_xyz, _art_art_dist_threshold):
                                        continue
                                    else:
                                        already_exists = True
                                        break

                                if not already_exists:
                                    self.artefact_list.append(self.art_xyz)
                        else:
                            # Draw rectangle and label with warning color (red)
                            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            rospy.sleep(0.5)
                            # rospy.logwarn("Could not retrieve 3D coordinates.")
                
            # Publish processed image
            processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_pub_.publish(processed_msg)
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error: {e}")
            
            
    def is_artifact_far_enough(self , artefact, art_xyz, threshold):
        return math.hypot(artefact[0] - art_xyz[0], artefact[1] - art_xyz[1]) > threshold

    # Exploration planning thread #########################################################
    def exploration_loop(self):
        while not rospy.is_shutdown():
            self.main_loop()
            rospy.sleep(0.5)

                                        
        

    def handle_waiting_for_map(self):
        print ( 'waiting for map.....')
        while self.current_map_ is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(1.0)  # Wait for the map to be received
            continue  # Keep waiting until the map is received
        # load the current map
        current_map = self.current_map_
        print ( ' map acquired')  
        self.exploration_state_ = ExplorationsState.IDENTIFYING_FRONTIERS
        print ( 'state changed to identifying frontiers')
    

    def handle_identifying_frontiers(self):
        print ( 'identifying frontiers,,,,,')
        frontiers = self.identify_frontiers(self.current_map_)
        if not frontiers:
            rospy.loginfo('No frontiers found!')
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
        else:
            self.exploration_state_ = ExplorationsState.SELECTING_FRONTIER
            

    def handle_selecting_frontier(self):
        print ( ' frontiers found selecting frontiers')
        self.selected_frontier = None
        frontiers = self.identify_frontiers(self.current_map_)
        rospy.loginfo(f'Frontiers: {len(frontiers)}')
        rospy.loginfo(f'Frontiers after min dist select: {len(frontiers)}')
        self.selected_frontier = self.select_nearest_frontier(frontiers)
        if self.selected_frontier is None:
            rospy.logwarn('No frontier selected.')
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
        else:
            
            rospy.loginfo('Frontier selected')
            self.exploration_state_ = ExplorationsState.MOVING_TO_FRONTIER



            
            
    def handle_timeout(self):
        rospy.loginfo("Timeout reached.")
        self.current_map_ = None
        self.selected_frontier = None
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        
                                        
    def identify_frontiers(self, current_map): 
        frontiers = []
       # Extract map dimensions and data
        width =current_map.info.width
        height =current_map.info.height
        data =current_map.data  # Occupancy grid data
        origin = current_map.info.origin.position
        resolution = current_map.info.resolution
        
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
        # rospy.sleep(0.0000000001)
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
                if self.compute_distance_between_points(random_frontier, frontier) < threshold_distance:
                    close_frontiers.append(frontier)
                    frontiers.remove(frontier)

            # Merge the close frontiers into one
            if close_frontiers:
                avg_x = sum(f[0] for f in close_frontiers) / len(close_frontiers)
                avg_y = sum(f[1] for f in close_frontiers) / len(close_frontiers)
                merged_frontiers.append([avg_x, avg_y])

        return merged_frontiers
        
        
    def compute_distance(self, frontier):
        map_res = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        x, y = frontier
        frontier_x = x * map_res + map_origin.x
        frontier_y = y * map_res + map_origin.y
        distance = np.hypot((frontier_x - self.current_position.x ), (frontier_y - self.current_position.y ))
        return distance
        
        
    def is_valid_frontier(self, frontier_point):
        # Check if the frontier point is far enough from all visited frontiers.
        min_distance = 0 # Define a minimum distance threshold
        
        for visited_point in self.visited_frontiers:
            # Calculate Euclidean distance between the frontier and visited point
            distance = np.hypot((frontier_point[0] - visited_point[0]), (frontier_point[1] - visited_point[1]))
            if distance < min_distance:
                return False  # The frontier is too close to a visited point
            
        return True
        
    def index_to_position(self, index):
        # Convert the index of the grid cell to a (x, y) position in the map
        width = self.current_map_.info.width
        resolution = self.current_map_.info.resolution
        origin_x = self.current_map_.info.origin.position.x
        origin_y = self.current_map_.info.origin.position.y
        
        x = (index % width) * resolution + origin_x
        y = (index // width) * resolution + origin_y

        
        return Point(x, y, 0)  # Return as a Point object
    def wrap_angle(self , angle):
        # Function to wrap an angle between 0 and 2*Pi
        while angle < 0.0:
            angle = angle + 2 * math.pi

        while angle > 2 * math.pi:
            angle = angle - 2 * math.pi

        return angle


    def pose2d_to_pose(self ,pose_2d):
        pose = Pose()

        pose.position.x = pose_2d.x
        pose.position.y = pose_2d.y

        pose.orientation.w = math.cos(pose_2d.theta / 2.0)
        pose.orientation.z = math.sin(pose_2d.theta / 2.0)

        return pose


    def compute_distance_between_points(self , point1, point2):
        return np.hypot((point1[0] - point2[0]), (point1[1] - point2[1]))
    
           
    
    def move_to_frontier(self, frontier):
        map_resolution = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        # send first selected frontier to move_base and remove it from
        # till all the frontiers are sent to move_base
        x , y = frontier
            # Send a goal to move_base to explore the selected frontier
        pose_2d = Pose2D
            # Move forward 10m
        pose_2d.x = x*map_resolution +map_origin.x
        pose_2d.y = y * map_resolution + map_origin.y
        pose_2d.theta = math.pi/2
        print (f'x:{pose_2d.x} , y:{pose_2d.y}')

        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ = self.goal_counter_ + 1
        action_goal.goal.target_pose.pose = self.pose2d_to_pose(pose_2d)
            
        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)
 
 
    def handle_moving_to_frontier(self, action_state):
        
         # Ensure there is a selected frontier to attempt
        if not self.selected_frontier:
            rospy.loginfo("No frontiers to move to.")
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
            return
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
                    self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
                    return  # Stop processing further frontiers in this call

                # Check if the goal was rejected or aborted
                elif action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                    rospy.loginfo("Goal rejected or aborted.")
                    self.exploration_state_ = ExplorationsState.HANDLE_REJECTED_FRONTIER
                    break  # Break out of the inner loop to attempt the next frontier
                
                # Check if the object has been identified
                elif self.exploration_state_ == ExplorationsState.OBJECT_IDENTIFIED_SCAN:
                    rospy.loginfo("Object identified, stopping exploration.")
                    return
                                            
                # If the loop ends due to timeout, handle the timeout case
            if rospy.Time.now() - start_time >= rospy.Duration(15):
                rospy.loginfo("Timeout reached.")
                self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
                break  # Stop processing further frontiers in this call

            # Check if no more frontiers are available
        if not self.selected_frontier:
            rospy.loginfo("No more frontiers to explore.")
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
                    
        return
       
       
    def object_identified_scan(self, action_state):
        # Ensure the object coordinates (x, y, theta) are initialized
        x_obj = None
        y_obj = None
        theta_obj = None
        delta_x = float('inf')
        delta_y = float('inf')
        delta_theta = float('inf')

        # Check if detected object coordinates are available
        if self.art_xyz is not None:
            # Extract the object coordinates
            x_obj = self.art_xyz[0]
            y_obj = self.art_xyz[1]
            theta_obj = math.atan2(y_obj, x_obj)  # Default orientation, adjust if necessary

            # Log and print the identified object position
            print(f"[DEBUG] Object found at x: {x_obj}, y: {y_obj}, theta: {theta_obj} ----------")
            
            # Adding tolerance to the object position
            pos_tolerance = 0.5
            theta_tolerance = 0.3
            print(f"[DEBUG] Tolerance levels -> Position: {pos_tolerance}, Theta: {theta_tolerance}")

            # Calculate the difference from the previous goal if available
            if hasattr(self, 'last_goal') and self.last_goal is not None:
                delta_x = abs(x_obj - self.last_goal[0])
                delta_y = abs(y_obj - self.last_goal[1])
                delta_theta = abs(theta_obj - self.last_goal[2])
                print(f"[DEBUG] Delta values -> Delta_x: {delta_x}, Delta_y: {delta_y}, Delta_theta: {delta_theta}")
            else:
                print("[DEBUG] No previous goal available to calculate deltas.")

            # If the difference is within tolerance, do not send a new goal
            if delta_x < pos_tolerance and delta_y < pos_tolerance and delta_theta < theta_tolerance:
                rospy.loginfo("[DEBUG] Object position change within tolerance, not sending a new goal.")
                self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
                return

            # Create and populate a MoveBaseGoal message to move to the identified object
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal.target_pose.header.stamp = rospy.Time.now()

            # Set the goal's position and orientation
            action_goal.goal.target_pose.pose.position.x = x_obj
            action_goal.goal.target_pose.pose.position.y = y_obj
            action_goal.goal.target_pose.pose.position.z = 0.0  # Since we're working in 2D

            # Calculate the orientation quaternion from theta
            quaternion = tf.transformations.quaternion_from_euler(0, 0, theta_obj)
            action_goal.goal.target_pose.pose.orientation = Quaternion(*quaternion)

            # Send the goal to move_base
            rospy.loginfo("[DEBUG] Sending goal to move_base with coordinates (x: {}, y: {}, theta: {})".format(
                x_obj, y_obj, theta_obj))
            self.move_base_action_client_.send_goal(action_goal.goal)
            
            # Save the current goal as the last goal
            self.last_goal = (x_obj, y_obj, theta_obj)
            print(f"[DEBUG] Last goal saved as -> x: {x_obj}, y: {y_obj}, theta: {theta_obj}")

            # Wait for the goal to be executed and check the state
            goal_rejected_threshold = rospy.Duration(5)  # Define how long to wait before checking if the goal is rejected
            self.move_base_action_client_.wait_for_result(goal_rejected_threshold)
            current_state = self.move_base_action_client_.get_state()

            # Handle goal success or failure based on the current state
            if current_state == actionlib.GoalStatus.SUCCEEDED:
                rospy.loginfo("Successfully reached the object!")
                self.exploration_state_ = ExplorationsState.MOVING_TO_FRONTIER
            elif current_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                rospy.loginfo("[DEBUG] Goal rejected or aborted. Cancelling goal and resuming exploration.")
                self.move_base_action_client_.cancel_goal()
                self.last_goal = None  # Clear the last goal
                self.exploration_state_ = ExplorationsState.MOVING_TO_FRONTIER

            # Small sleep to ensure actions are processed
            rospy.sleep(2)
        else:
            # Handle the case where no object is detected
            rospy.logwarn('[DEBUG] Object not found')
            self.exploration_state_ = ExplorationsState.MOVING_TO_FRONTIER
            rospy.sleep(2)

    def exploration_planner(self, action_state):
        if action_state != actionlib.GoalStatus.ACTIVE:
            print('Exploration planner ............')
            if self.exploration_state_ == ExplorationsState.WAITING_FOR_MAP:
                print ( 'enteringg .... waiting for map')
                self.handle_waiting_for_map()

            elif self.exploration_state_ == ExplorationsState.IDENTIFYING_FRONTIERS:
                print ( 'entering ....IDENTifying frontiers')
                self.handle_identifying_frontiers()

            elif self.exploration_state_ == ExplorationsState.SELECTING_FRONTIER:
                print ( 'entering ....SELECTING frontiers')
                self.handle_selecting_frontier()

            elif self.exploration_state_ == ExplorationsState.MOVING_TO_FRONTIER:
                print ( 'entering ....MOVING TO frontiers')
                self.handle_moving_to_frontier(action_state)

            elif self.exploration_state_ == ExplorationsState.HANDLE_REJECTED_FRONTIER:
                print ( 'entering ....HANDLE REJECTED frontiers')
                self.handle_selecting_frontier()
                
            elif self.exploration_state_ == ExplorationsState.HANDLE_TIMEOUT:
                print ( 'entering ....HANDLE TIMEOUT')
                self.handle_timeout()

            elif self.exploration_state_ == ExplorationsState.EXPLORED_MAP:
                rospy.loginfo("Exploration completed successfully.")
                
            elif self.exploration_state_ == ExplorationsState.OBJECT_IDENTIFIED_SCAN:
                rospy.loginfo("Object planner started checkkiiiiiiiiiiiiiiiiiiiiiiing ............")
                self.object_identified_scan(action_state )

       
    def main_loop(self):

        while not rospy.is_shutdown():

            #######################################################
            # Get the current status
            # See the possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number:' + str(action_state))
            
            if (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully explored!")
                # if frontiers 
                #     self.exploration_done_ = True
            elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE):
                print("Exploration preempted!")
                self.exploration_done_ = False
                action_state = actionlib.GoalStatus.PREEMPTING
            elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE) and (self.exploration_state_ == ExplorationsState.OBJECT_IDENTIFIED_SCAN):
                print (' moving to object  ')
                self.exploration_done_= False
                ## put in change state and call object scan 

            #######################################################
            # Select the next planner to execute - Update this logic as you see fit!
            if not self.exploration_done_:
                self.planner_type_ = PlannerType.EXPLORATION

            #######################################################
            # Execute the planner by calling the relevant method - methods send a goal to "move_base" with "self.move_base_action_client_"
            # Add your own planners here!
            print("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.EXPLORATION: 
                self.exploration_planner(action_state)

            #######################################################
            # Delay so the loop doesn't run too fast
            rospy.sleep(0.5)



if __name__ == '__main__':
    rospy.init_node('cave_explorer')

    # Create the cave explorer instance
    cave_explorer = CaveExplorer()

    # Keep the node running
    rospy.spin()
