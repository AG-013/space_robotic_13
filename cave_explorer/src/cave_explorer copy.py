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
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
from visualization_msgs.msg import Marker
import rospy
import roslib


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
        self.image_pub_ = rospy.Publisher("detections_image", Image, queue_size=5)

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
        # self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ \
        #                         SE3.RPY(1.57, 0, 1.57) @ \
        #                         SE3(0.005, 0.028, 0.013) @ \
        #                         SE3.RPY(-1.57, 0, -1.57)
        self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ SE3(0.005, 0.028, 0.013)
        ##################################################################
        
        self.marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)
        
        
    
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
        # print('printing robot position')
        # print(robot_pos)
        # print('-------------------------------\n')
        
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
            # print(point_transform)
            
            # # Debug prints
            # rospy.logdebug('\n------------------------------------------')
            # rospy.logdebug(f'Robot Position:\n{robot_pos}')
            # rospy.logdebug(f'Camera Transform:\n{self.base_2_depth_cam}')
            # rospy.logdebug(f'Point Before Transformation:\n{point_transform}')
            
            # Apply transformations in correct order
            # First transform from camera optical frame to robot base frame
            # point_in_base = self.base_2_depth_cam @ point_transform
            
            # # Then transform from robot base to world frame
            # point_in_world = robot_pos @ point_in_base
            
            point_in_world = robot_pos @ self.base_2_depth_cam @ point_transform
            # print('point in world')
            # print(point_in_world)
            
            # rospy.logdebug(f'Final Point Position:\n{point_in_world}')
            # rospy.logdebug('------------------------------------------\n')
            
            # Extract the translation components (t) from the final transformation
            return point_in_world.t # .t extract translational component

        
        return None


    def depth_callback(self, depth_msg):
        try:
            self.depth_data_ = depth_msg
        except Exception as e:
            rospy.logwarn(f"Error in depth callback: {e}")
            
            
    def publish_marker(self, x, y, z, isMineral):
        marker = Marker()
        marker.header.frame_id = "map"  # Replace with your frame ID
        marker.header.stamp = rospy.Time.now()
        marker.ns = "artifact_marker"
        marker.type = Marker.SPHERE  # Or any other shape
        marker.action = Marker.ADD
        marker.pose.position.x = x  # 3D coordinates from the bounding box center
        marker.pose.position.y = y
        marker.pose.position.z = z
        marker.scale.x = 1  # Size of the marker (adjust as needed)
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0  # Alpha value (opacity)
        if isMineral:
            marker.color.r = 0.004
            marker.color.g = 0.195
            marker.color.b = 0.125
        else:         # Else it is grey
            marker.color.r = 0.5
            marker.color.g = 0.5
            marker.color.b = 0.5
    
        marker.lifetime = rospy.Duration(0.1)  # 0 means forever, or set a specific duration in seconds
                            
        self.marker_pub.publish(marker)
        # print(f'PLACED MARKER at: {x}, {y}, {z}')


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
                            # self.go_to_artifact()
                            
                            # Add RViz Marker for visualization
                            self.publish_marker(art_xyz[0], art_xyz[1], art_xyz[2], (class_id == 1))
    
                    else:
                        # Draw rectangle and label
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        rospy.logwarn("Could not retrieve 3D coordinates.")

            
            # print('\n\n\n')

        # Convert the modified CV2 image back to a ROS Image message
        try:
            processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Publish the processed image
        self.image_pub_.publish(processed_msg)
        # rospy.loginfo("Published processed image")


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