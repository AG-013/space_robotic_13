#!/usr/bin/env python3

# import rospy
# import roslib
# import math
# import cv2 # OpenCV2
# from cv_bridge import CvBridge, CvBridgeError
# import numpy as np
# from nav_msgs.srv import GetMap
# from nav_msgs.msg import OccupancyGrid
# from nav_msgs.msg import Odometry
# import tf
# from std_srvs.srv import Empty
# from sensor_msgs.msg import LaserScan
# from geometry_msgs.msg import Twist
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from geometry_msgs.msg import Pose2D
# from geometry_msgs.msg import Pose
# from geometry_msgs.msg import Point
# from sensor_msgs.msg import Image
# from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
# import actionlib
# import random
# import copy
# from threading import Lock
# from enum import Enum

import os
import random
import copy
from threading import Lock
from enum import Enum
import time

# Math Modules
from spatialmath import SE3
# May have to install these packages:
# pip install roboticstoolbox-python
# pip install spatialmath-python
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
import threading



def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    EXPLORATION = 6
    # Add more!
    
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
        # Variables/Flags for perception
        self.localised_ = False
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.goal_counter_ = 0  # Unique ID for each goal sent to move_base
        self.exploration_done_ = False
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        self.visited_frontiers = set()

        # Initialise CvBridge for image processing
        self.cv_bridge_ = CvBridge()

        # Wait for the transform from map to base_link
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        # Publishers
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)  # Manual control
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)  # Camera detections
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)  # Artifact markers

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Subscribers
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.depth_sub_ = rospy.Subscriber("/camera/depth/points", PointCloud2, self.depth_callback, queue_size=1)
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.odom_sub_ = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.laser_sub_ = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)

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
        self.scan_data = None

        # Artifact storage
        self.mineral_artefacts = []
        self.mushroom_artefacts = []

        # Set confidence threshold for object detection
        self.confidence_threshold = 0.5

        # Timer to publish artifact markers
        self.marker_timer = rospy.Timer(rospy.Duration(0.5), self.publish_artefact_markers)


    def depth_callback(self, depth_msg):
        try:
            self.depth_data_ = depth_msg
        except Exception as e:
            rospy.logwarn(f"Error in depth callback: {e}")

        # map service
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
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        # print("pose: ", pose)

        return pose

    def create_marker(self, art_xyz, marker_id, r, g, b):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "artifact_marker"
        marker.id = marker_id
        
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = art_xyz[0]
        marker.pose.position.y = art_xyz[1]
        marker.pose.position.z = art_xyz[2]
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.lifetime = rospy.Duration(0.6)
        
        # Add orientation (required for proper visualization)
        marker.pose.orientation.w = 1.0
        
        return marker

    def publish_artefact_markers(self, event):
        marker_id = 0  # Start counter for unique IDs
        
        # Publish mineral markers (green)
        for art_xyz in self.mineral_artefacts:
            marker = self.create_marker(art_xyz, marker_id, 0.004, 0.195, 0.125)
            self.marker_pub.publish(marker)
            marker_id += 1  # Increment for the next marker

        # Publish mushroom markers (grey)
        for art_xyz in self.mushroom_artefacts:
            marker = self.create_marker(art_xyz, marker_id, 0.5, 0.5, 0.5)
            self.marker_pub.publish(marker)
            marker_id += 1  # Increment for the next marker


    def image_callback(self, image_msg):
        # object identified
        # self.exploration_planner(ExplorationsState.OBJECT_IDENTIFIED_SCAN)
        
        classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]

        try:
            # Convert the ROS image message to a CV2 image
            cv_image = self.cv_bridge_.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Process the image using YOLO
        results = self.model_(cv_image, device=self.device_, imgsz=(480, 384))
        

        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0].item()  # Confidence score

                # Only process boxes with confidence above the threshold
                if confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls.item())  # Get the class ID from the tensor
                    label = f'{classes[class_id]} {confidence:.2f}'  # Class name and confidence

                    # Calculate and print center point of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    rospy.loginfo(f"Center of {classes[class_id]}: ({center_x}, {center_y})")

                    # Get the 3D coordinates
                    art_xyz = self.get_posed_3d(center_x, center_y)
                    
                    # Check if art_xyz is None before accessing its elements
                    if art_xyz is not None:
                        # rospy.loginfo(f"X: {art_xyz[0]},  Y: {art_xyz[1]},  Z: {art_xyz[2]}")
                        print(art_xyz)
                        
                        # Draw rectangle and label
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

                        # classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]
                        # Detecting Mineral and Mushroom - choosing mushroom because mineral and mushroom aren't together, so less points of error                         
                        if class_id == 1 or class_id == 4:
                            if class_id == 1:
                                artefact_list = self.mineral_artefacts
                            else:
                                artefact_list = self.mushroom_artefacts
                            # Check if it doesn't already exist first
                            already_exists = False
                            for artefact in artefact_list:
                                
                                # When having multiple markers make sure that they are at least a min distance away from each other
                                _art_art_dist_treshold = 7
        
                                if math.hypot(artefact[0] - art_xyz[0], artefact[1] - art_xyz[1]) > _art_art_dist_treshold:
                                    continue
                                else:
                                    already_exists = True
                                    break
                            if not already_exists:
                                artefact_list.append(art_xyz)
                                # self.go_to_artifact(art_xyz)
                    else:
                        # Draw rectangle and label
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        rospy.logwarn("Could not retrieve 3D coordinates.")

        # Convert the modified CV2 image back to a ROS Image message
        try:
            processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Publish the processed image
        self.image_pub_.publish(processed_msg)
        # _--------------------------------------------------------------


    def go_to_artifact(self, artefact_trans: tuple):
            # Navigate robot to target artifact position

            # Constants
            ANGLE_THRESHOLD = 0.1  # Radians
            DISTANCE_THRESHOLD = 1.75  # m
            LINEAR_SPEED = 0.5  # m/s
            ANGULAR_SPEED = 0.5  # rad/s
            REVERSE_DISTANCE = 1.0  # m to reverse after reaching the artifact

            # 1. Initial Rotation to face target
            pose_2d = self.get_pose_2d()
            target_theta = math.atan2(artefact_trans[1] - pose_2d.y, artefact_trans[0] - pose_2d.x)
            cur_theta = wrap_angle(pose_2d.theta)

            # Store the original angle before rotation to face the artifact
            original_theta = cur_theta

            while abs(wrap_angle(target_theta - cur_theta)) > ANGLE_THRESHOLD:
                rospy.logerr('Rotating Robot to face artifact')
                pose_2d = self.get_pose_2d()
                cur_theta = wrap_angle(pose_2d.theta)
                
                twist_msg = Twist()
                angle_diff = wrap_angle(target_theta - cur_theta)
                twist_msg.angular.z = ANGULAR_SPEED * angle_diff
                
                self.cmd_vel_pub_.publish(twist_msg)
                rospy.sleep(0.1)

            # Stop rotation
            self.cmd_vel_pub_.publish(Twist())
            rospy.sleep(0.5)  # Brief pause after rotation

            # 2. Forward Movement
            pose_2d = self.get_pose_2d()
            distance = self.get_distance_to_target(artefact_trans, pose_2d)

            while distance > DISTANCE_THRESHOLD and not rospy.is_shutdown():
                while self.about_to_collide():
                    # Avoid obstacles while moving toward the artifact
                    twist_msg = Twist()
                    if self.move_left():
                        rospy.logerr(f'ABOUT TO COLLIDE SO MOVING LEFT: {distance:.2f}m')
                        twist_msg.linear.y = 0.5
                    else:
                        rospy.logerr(f'ABOUT TO COLLIDE SO MOVING RIGHT: {distance:.2f}m')
                        twist_msg.linear.y = -0.5
                    self.cmd_vel_pub_.publish(twist_msg)
                
                rospy.logerr(f'Moving forward, distance: {distance:.2f}m')
                pose_2d = self.get_pose_2d()
                
                # Recalculate target angle and current angle difference
                target_theta = math.atan2(artefact_trans[1] - pose_2d.y, artefact_trans[0] - pose_2d.x)
                angle_diff = wrap_angle(target_theta - wrap_angle(pose_2d.theta))
                
                twist_msg = Twist()
                twist_msg.linear.x = LINEAR_SPEED
                twist_msg.angular.z = 0.5 * ANGULAR_SPEED * angle_diff
                
                self.cmd_vel_pub_.publish(twist_msg)
                
                # Update distance
                distance = self.get_distance_to_target(artefact_trans, pose_2d)
                rospy.sleep(0.1)

            # Stop the robot after reaching the artifact
            self.cmd_vel_pub_.publish(Twist())
            rospy.sleep(2)
            # rospy.loginfo('Arrived at the artifact')
            rospy.loginfo("\033[93mArrived at the artifact\033[0m")

            # 3. Reverse movement
            rospy.loginfo('Reversing away from the artifact')
            reverse_distance = 0
            while reverse_distance < REVERSE_DISTANCE and not rospy.is_shutdown():
                twist_msg = Twist()
                twist_msg.linear.x = -LINEAR_SPEED  # Move backward
                self.cmd_vel_pub_.publish(twist_msg)
                
                rospy.sleep(0.1)
                reverse_distance += LINEAR_SPEED * 0.1  # Approximate how much the robot has moved

            # Stop the robot after reversing
            self.cmd_vel_pub_.publish(Twist())
            rospy.loginfo('Reversed away from the artifact')

            # 4. Rotate back to original orientation
            rospy.loginfo('Rotating back to the original orientation')
            cur_theta = wrap_angle(self.get_pose_2d().theta)
            while abs(wrap_angle(original_theta - cur_theta)) > ANGLE_THRESHOLD and not rospy.is_shutdown():
                cur_theta = wrap_angle(self.get_pose_2d().theta)
                
                twist_msg = Twist()
                angle_diff = -wrap_angle(original_theta - cur_theta)
                twist_msg.angular.z = ANGULAR_SPEED * angle_diff
                self.cmd_vel_pub_.publish(twist_msg)
                rospy.sleep(0.1)

            # Stop rotation
            self.cmd_vel_pub_.publish(Twist())
            rospy.loginfo('Returned to original orientation')

 
    def about_to_collide(self):
        # Check if the robot is about to collide based on the latest laser scan data.
        SAFETY_DISTANCE = 0.85  # Set safety distance threshold (in meters)
        
        # Ensure thread-safe access to laser scan data
        with self.scan_lock:
            if self.scan_data is None:
                rospy.logwarn("Laser scan data is not available")
                return False  # No data, assume no collision risk

            # Determine the forward sector to check (e.g., front 60 degrees)
            # Get the indices for the front sector (centered in the middle of the scan data)
            # Extract the relevant ranges for the front sector
            front_ranges_left = self.scan_data[0:60]
            front_ranges_right = self.scan_data[300:]
            front_ranges = front_ranges_left + front_ranges_right
            valid_ranges = [r for r in front_ranges if not math.isinf(r) and not math.isnan(r)]

            if not valid_ranges:
                rospy.logwarn("No valid laser scan data in the forward sector")
                return False  # No valid data, assume no collision risk

            # Check if any obstacle is within the safety distance
            if min(valid_ranges) < SAFETY_DISTANCE:
                rospy.logerr("Obstacle detected within safety distance!")
                return True  # Collision likely
            else:
                rospy.logerr(f"NO Obstacle detected SAFE TO MOVE! {min(valid_ranges)}")
                return False  # No collision risk
    
    
    # Checks if robot can turn left or right                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    def move_left(self):
        # Check if the robot is about to collide based on the latest laser scan data.
        SAFETY_DISTANCE = 1.0  # Set safety distance threshold (in meters)
        
        # Ensure thread-safe access to laser scan data
        with self.scan_lock:
            if self.scan_data is None:
                rospy.logwarn("Laser scan data is not available")
                return False  # No data, assume no collision risk

            # Determine the forward sector to check (e.g., front 60 degrees)
            # Get the indices for the front sector (centered in the middle of the scan data)
            # Extract the relevant ranges for the front sector
            ranges_left = self.scan_data[60:120]
            valid_ranges = [r for r in ranges_left if not math.isinf(r) and not math.isnan(r)]

            if not valid_ranges:
                rospy.logwarn("move_left(): No valid laser scan data in the left sector, so just gonna turn right")
                return False  # No valid data, assume no collision risk

            # Check if any obstacle is within the safety distance
            if min(valid_ranges) < SAFETY_DISTANCE:
                rospy.logerr("move_left(): Obstacle detected within safety distance!, so just gonna turn right")
                return False  # Collision likely
            else:
                rospy.logerr(f"move_left(): NO Obstacle detected SAFE TO MOVE LEFT! {min(valid_ranges)}")
                return True  # No collision risk

# 5555555555555555555555555555555555555555555555555555555555555555555555555555555555555

    def map_callback(self, map_msg):
        if self.current_map_ is None:
            rospy.loginfo("Map received")
            self.current_map_ = map_msg  
        elif self.current_map_.header.stamp != map_msg.header.stamp:
            rospy.loginfo("Map updated")
            self.current_map_ = map_msg
        else:
            rospy.loginfo("Map not updated")
                    
        
    def odom_callback(self,odom_sub_):
        # Extract position data
        # rospy.loginfo ("Odo received")
        self.odom_ = odom_sub_

        self.current_position = Point()

        # Extract pose
        position = self.odom_.pose.pose.position
        self.current_position.x = position.x
        self.current_position.y = position.y

        orientation = self.odom_.pose.pose.orientation
        # Extract twist 
        linear = self.odom_.twist.twist.linear
        angular = self.odom_.twist.twist.angular
        
        
    def laser_callback(self,laser_sub_):
        self.laser_scan_ = laser_sub_
        # Extract laser scan data
        rospy.loginfo ("Laser received")
        self.scan_ranges = self.laser_scan_.ranges
        
    
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
                rospy.loginfo("Object identified.")
                self.object_identified_scan()
                
                
    def object_identified_scan(self):
        # Stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub_.publish(twist)
        
        # identify object coordinates and move towards it
        # align so that the object is in the center of the camera
        # move towards the object and 
        


        # Return to the exploration state
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP

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

    def handle_moving_to_frontier(self, action_state):
        
        ## as the robot moves to the frontier, it will check object detection
        ## if object detected, it will stop and move towards the object
        
        ## implement logic for changing planner to object detection
        
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
                    self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
                    return  # Stop processing further frontiers in this call

                # Check if the goal was rejected or aborted
                elif action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                    rospy.loginfo("Goal rejected or aborted.")
                    self.exploration_state_ = ExplorationsState.HANDLE_REJECTED_FRONTIER
                    break  # Break out of the inner loop to attempt the next frontier
                
            # If the loop ends due to timeout, handle the timeout case
            if rospy.Time.now() - start_time >= rospy.Duration(15):
                rospy.loginfo("Timeout reached.")
                self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
                break  # Stop processing further frontiers in this call

        # Check if no more frontiers are available
        if not self.selected_frontier:
            rospy.loginfo("No more frontiers to explore.")
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
            
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
        return neighbors
    
    
    def select_nearest_frontier(self, frontiers):
        frontiers_merged = frontiers
            
        # rospy.loginfo(f'Frontiers found: {len(frontiers)}')
        frontiers_merged= self.merge_close_frontiers_to_one(frontiers)
        
        ### use a weighted frontier selection process 
         ### based on distance alpha 1 , - size of frontier alpha 2, + orientation towards the frontier alpha 3
         ## alpha 1 = 0.5, alpha 2 = 0.3, alpha 3 = 0.2
        
        
        # rospy.loginfo(f'Frontiers merged: {len(frontiers_merged)}')
        # frontiers_merged.sort(key=self.compute_distance)
        # print ( f'frontiers sorted distance: {len(frontiers_merged)}')
        # rospy.loginfo(f'Frontiers sorted: {len(frontiers_merged)}')
        # frontiers_merged =self.is_valid_frontier(frontiers_merged)
        # print ( f'frontiers sorted distance + is valid: {len(frontiers_merged)}')
        # print ( 'frontiers sorted')
        return frontiers_merged
    
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

    def compute_distance_between_points(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        
    def compute_distance(self, frontier):
        map_res = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        x, y = frontier
        frontier_x = x * map_res + map_origin.x
        frontier_y = y * map_res + map_origin.y
        distance = np.sqrt((frontier_x - self.current_position.x ) ** 2 + (frontier_y - self.current_position.y ) ** 2)
        return distance
        
    def is_valid_frontier(self, frontier_point):
        # """ Check if the frontier point is far enough from all visited frontiers. """
        min_distance = 0 # Define a minimum distance threshold
        
        for visited_point in self.visited_frontiers:
            # Calculate Euclidean distance between the frontier and visited point
            distance = np.sqrt((frontier_point[0] - visited_point[0]) ** 2 + 
                            (frontier_point[1] - visited_point[1]) ** 2)
            if distance < min_distance:
                return False  # The frontier is too close to a visited point
        
        return True
        
        
    
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
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)
            
            # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)
            
        
    def index_to_position(self, index):
        # Convert the index of the grid cell to a (x, y) position in the map
        width = self.current_map_.info.width
        resolution = self.current_map_.info.resolution
        origin_x = self.current_map_.info.origin.position.x
        origin_y = self.current_map_.info.origin.position.y
        
        x = (index % width) * resolution + origin_x
        y = (index // width) * resolution + origin_y
        return Point(x, y, 0)  # Return as a Point object


  
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
                self.exploration_done_ = True
            elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE):
                print("Exploration preempted!")
                self.exploration_done_ = False
                action_state = actionlib.GoalStatus.PREEMPTING

            #######################################################
            # Select the next planner to execute
            # Update this logic as you see fi   t!
            if not self.exploration_done_:
                self.planner_type_ = PlannerType.EXPLORATION


            #######################################################
            # Execute the planner by calling the relevant method
            # The methods send a goal to "move_base" with "self.move_base_action_client_"
            # Add your own planners here!
            print("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.EXPLORATION: 
                self.exploration_planner(action_state)

            #######################################################
            # Delay so the loop doesn't run too fast
            rospy.sleep(0.2)



if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('cave_explorer')

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    # Loop forever while processing callbacks
    cave_explorer.main_loop()




