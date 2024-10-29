#!/usr/bin/env python3

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
from nav_msgs.msg import OccupancyGrid
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


class CaveExplorer:
    def __init__(self):

        # Variables/Flags for perception
        self.localised_ = False
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False
        self.goal_counter_ = 0 # gives each goal sent to move_base a unique ID

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Wait for the transform to become available
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            print("Waiting for transform... Have you launched a SLAM node?")        

        # Advertise "cmd_vel" publisher to control the robot manually -- though usually we will be controller via the following action client
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Publisher for the camera detections
        self.image_pub_ = rospy.Publisher("/detections_image", Image, queue_size=5)

        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        
        # self.depth_sub_ = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1)
        self.depth_sub_ = rospy.Subscriber("/camera/depth/points", PointCloud2, self.depth_callback, queue_size=1)

        # Initialize YOLO model
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"

        path = os.path.abspath(__file__)
        src_dir = os.path.dirname(path)
        parent_dir = os.path.abspath(os.path.join(src_dir, '..', '..'))
        model_path = os.path.join(parent_dir, 'cam_assist/src/test_train/yolov11s_trained_optimized.pt')
        self.model_ = YOLO(model_path)
        rospy.loginfo(f"Using device: {self.device_}")

        # Set confidence threshold
        confidence_threshold = 0.5
        self.confidence_threshold = confidence_threshold
        
        # For depth ############################################
        self.depth_data_= None
        ########################################################
                
        # For Transformation #############################################
        self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ SE3(0.005, 0.028, 0.013)
        ##################################################################
                        
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
        self.marker_timer = rospy.Timer(rospy.Duration(0.5), self.publish_artefact_markers)
        
        self.mineral_artefacts = []
        self.mushroom_artefacts = []
        
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)
        self.scan_lock = threading.Lock()
        self.scan_data = None   


    def laser_callback(self, scan_msg):
        """Callback function for laser scan data."""
        with self.scan_lock:
            self.scan_data = scan_msg.ranges
                   
        
    def publish_artefact_markers(self, event):
        # print('Number of Minerals', len(self.mineral_artefacts))
        # print('Number of Mushrooms', len(self.mushroom_artefacts))
        
        marker_id = 0  # Start counter for unique IDs
        
        # Publish mineral markers
        for art_xyz in self.mineral_artefacts:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "artifact_marker"
            marker.id = marker_id  # Assign unique ID
            marker_id += 1  # Increment for next marker
            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = art_xyz[0]
            marker.pose.position.y = art_xyz[1]
            marker.pose.position.z = art_xyz[2]
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.a = 1.0
            # Green colour for mineral
            marker.color.r = 0.004
            marker.color.g = 0.195
            marker.color.b = 0.125
            marker.lifetime = rospy.Duration(0.6)  # Increased slightly to ensure visibility
            
            # Add orientation (required for proper visualization)
            marker.pose.orientation.w = 1.0
            
            self.marker_pub.publish(marker)

        # Publish mushroom markers
        for art_xyz in self.mushroom_artefacts:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "artifact_marker"
            marker.id = marker_id  # Use next unique ID
            marker_id += 1
            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = art_xyz[0]
            marker.pose.position.y = art_xyz[1]
            marker.pose.position.z = art_xyz[2]
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.a = 1.0
            # Grey Colour for Mushroom
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
            marker.lifetime = rospy.Duration(0.6)
            marker.pose.orientation.w = 1.0
            
            self.marker_pub.publish(marker)
        
        
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

        return pose


    # Cons of this code: only does one pixel - Since depth data is noisy it is essential to get nearby pixels, and average them    
    # Given pix_xy (a tuple), extract the 3d position of the pixels by applying a transformation matrix of the current pose * matrix of relative pose    
    # Extract the 3D coordinates from PointCloud2 data at pixel (pixel_x, pixel_y)        
    def get_posed_3d(self, pixel_x: int, pixel_y: int) -> tuple:
        # Check if data is available
        if not self.depth_data_:
            rospy.logwarn("Depth message not received yet!")
            return None
        
        # Get current robot pose transformation
        (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
        x, y, z = trans
        qz = rot[2]
        qw = rot[3]
        
        theta = wrap_angle(2.0 * math.acos(qw) if qz >= 0 else -2.0 * math.acos(qw))
        
        # Create robot pose transformation
        robot_pos = SE3(x, y, z) @ SE3.Rz(theta)  # Using Rz for 2D rotation instead of RPY

        
        # Extract point from depth data
        point = list(point_cloud2.read_points(
            self.depth_data_, 
            field_names=("x", "y", "z"), 
            skip_nans=True, 
            uvs=[(pixel_x, pixel_y)]
        ))
        
        if point:
            # Convert point from camera frame to robot base frame
            old_x, old_y, old_z = point[0]
            
            # Transform from camera optical frame to camera frame
            x = old_z
            y = -old_x
            z = -old_y
            
            # Create point transformation
            point_transform = SE3.Trans(x, y, z)
            point_in_world = robot_pos @ self.base_2_depth_cam @ point_transform
            
            # Extract the translation components (t) from the final transformation
            return point_in_world.t # .t extract translational component

        return None


    def depth_callback(self, depth_msg):
        try:
            self.depth_data_ = depth_msg
        except Exception as e:
            rospy.logwarn(f"Error in depth callback: {e}")


    def get_distance_to_target(self, target_pos, current_pose):
        """Calculate Euclidean distance to target."""
        return math.sqrt( (target_pos[0] - current_pose.x) ** 2 + (target_pos[1] - current_pose.y) ** 2)


    def image_callback(self, image_msg):
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
                                self.go_to_artifact(art_xyz)
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
        # rospy.loginfo("Published processed image")
        
    
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
        
        
    # Simply move forward by 10m
    def planner_move_forwards(self, action_state):

        # Only send this once before another action
        if action_state == actionlib.GoalStatus.LOST:

            pose_2d = self.get_pose_2d()

            rospy.loginfo('Current pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            # Move forward 10m
            pose_2d.x += 10 * math.cos(pose_2d.theta)
            pose_2d.y += 10 * math.sin(pose_2d.theta)

            rospy.loginfo('Target pose: ' + str(pose_2d.x) + ' ' + str(pose_2d.y) + ' ' + str(pose_2d.theta))

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            # TEMP BLOCK
            # self.move_base_action_client_.send_goal(action_goal.goal)


    def planner_go_to_first_artifact(self, action_state):
        # Go to a pre-specified artifact (alien) location

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a pre-specified goal location
            pose_2d = Pose2D()
            pose_2d.x = 18.0
            pose_2d.y = 25.0
            pose_2d.theta = -math.pi/2

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            # TEMP BLOCK
            # self.move_base_action_client_.send_goal(action_goal.goal)


    # Go to the origin
    def planner_return_home(self, action_state):

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a pre-specified goal location
            pose_2d = Pose2D()
            pose_2d.x = 0
            pose_2d.y = 0
            pose_2d.theta = 0

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            # TEMP BLOCK
            # self.move_base_action_client_.send_goal(action_goal.goal)


    def planner_random_walk(self, action_state):
        # Go to a random location, which may be invalid

        min_x = -5
        max_x = 50
        min_y = -5
        max_y = 50

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a random location
            pose_2d = Pose2D()
            pose_2d.x = random.uniform(min_x, max_x)
            pose_2d.y = random.uniform(min_y, max_y)
            pose_2d.theta = random.uniform(0, 2*math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            # TEMP BLOCK
            # self.move_base_action_client_.send_goal(action_goal.goal)


    # Go to a random location out of a predefined set
    def planner_random_goal(self, action_state):
        
        # Hand picked set of goal locations
        random_goals = [[53.3, 40.7], [44.4, 13.3], [2.3, 33.4], [9.9, 37.3], [3.4, 18.5], [6.0, 0.4], [28.3, 11.8], [43.7, 12.8], [38.9, 43.0], [47.4, 4.7], [31.5, 3.2], [36.6, 32.5]]

        # Only send this if not already going to a goal
        if action_state != actionlib.GoalStatus.ACTIVE:

            # Select a random location
            idx = random.randint(0,len(random_goals)-1)
            pose_2d = Pose2D()
            pose_2d.x = random_goals[idx][0]
            pose_2d.y = random_goals[idx][1]
            pose_2d.theta = random.uniform(0, 2*math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id = self.goal_counter_
            self.goal_counter_ = self.goal_counter_ + 1
            action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

            rospy.loginfo('Sending goal...')
            # TEMP BLOCK
            # self.move_base_action_client_.send_goal(action_goal.goal)


    def main_loop(self):

        while not rospy.is_shutdown():
            #######################################################
            # Get the current status
            # See the possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number:' + str(action_state))

            if (self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                print("Successfully reached first artifact!")
                self.reached_first_artifact_ = True
                print("Successfully returned home!")
                self.returned_home_ = True

            #######################################################
            # Select the next planner to execute
            # Update this logic as you see fit!
            # self.planner_type_ = PlannerType.MOVE_FORWARDS
            if not self.reached_first_artifact_:
                self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
            elif not self.returned_home_:
                self.planner_type_ = PlannerType.RETURN_HOME
            else:
                self.planner_type_ = PlannerType.RANDOM_GOAL

            #######################################################
            # Execute the planner by calling the relevant method
            # The methods send a goal to "move_base" with "self.move_base_action_client_"
            # Add your own planners here!
            print("Calling planner:", self.planner_type_.name)
            if self.planner_type_ == PlannerType.MOVE_FORWARDS:
                self.planner_move_forwards(action_state)
            elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
                self.planner_go_to_first_artifact(action_state)
            elif self.planner_type_ == PlannerType.RETURN_HOME:
                self.planner_return_home(action_state)
            elif self.planner_type_ == PlannerType.RANDOM_WALK:
                self.planner_random_walk(action_state)
            elif self.planner_type_ == PlannerType.RANDOM_GOAL:
                self.planner_random_goal(action_state)

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