#!/usr/bin/env python3

# Math Modules
# May have to install these packages:
# pip install roboticstoolbox-python
# pip install spatialmath-python
import math
import numpy as np
import random
from sklearn.cluster import DBSCAN

# ROS Modules
import tf
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D
import os
import rospy
import roslib
import math
import cv2 # OpenCV2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import random
import copy
from threading import Lock
from enum import Enum
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point



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

from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal, MoveBaseGoal
import actionlib
import rospy
from std_srvs.srv import Trigger, TriggerResponse  # Import the Trigger service for simplicity

from helper_functions import *
from enums import *
class Node:
    def __init__(self, x, y, idx, link):

        # Index of the node in the graph
        self.idx = idx

        # Position of node
        self.x = x
        self.y = y

        self.link = link


class CaveExplorer:
    
    TIME_OUT_MAX = 27.5
    
    def __init__(self):
        rospy.init_node('cave_explorer', anonymous=True)

        self.MAP_WIDTH = 896
        self.MAP_HEIGHT = 896
        self.MIN_CLUSTER_POINTS = 50
        self.INTENSITY_THRESHOLD = 10
        self.LENGTH_WEIGHT = 5.0
        self.DIST_WEIGHT = 2.2

        self.occupancy_grid = None
        self.goal_counter_ = 0
        self.exploration_state_ = PlannerType.WAITING_FOR_MAP

        self.chosen_frontier_pose = None
        
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        # Subscribers
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")        
        
        # Callbacks
        rospy.wait_for_service('get_artifact_location')
        self.artifact_location_service_client = rospy.ServiceProxy('get_artifact_location', Trigger)
        self.get_artifact_location()
        self.artifact_check_timer = rospy.Timer(rospy.Duration(2.0), self.timer_artifact_callback)
        self.artefact_x_y = None
        self.artefacts_list = []


    # Callback Functions ##########################################################################################
    def map_callback(self, msg):
        self.occupancy_grid = msg.data
        rospy.sleep(0.2)

    def get_artifact_location(self):
        try:
            if self.exploration_state_ != PlannerType.OBJECT_IDENTIFIED_SCAN:
                response = self.artifact_location_service_client()
                if response.success:
                    try:
                        x, y, z = map(float, response.message.split(','))
                        return (x, y, z)
                    except ValueError as e:
                        rospy.logerr(f"Failed to parse coordinates: {str(e)}")
                        return None
                else:
                    rospy.logwarn(response.message)
                    return None
            else:
                rospy.logerr('NOT ACCEPTING NEW ARTEFACTS AS IM BUSY GOING TO ONE')
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {str(e)}")
            return None

    def timer_artifact_callback(self, event):
        coords = self.get_artifact_location()
        if coords is not None:
            if coords not in self.artefacts_list:
                self.artefacts_list.append(coords)
                self.artefact_x_y = coords
                rospy.loginfo(f"NEW: Found artifact at X: {self.artefact_x_y[0]:.2f}, Y: {self.artefact_x_y[1]:.2f}, Theta: {self.artefact_x_y[2]:.2f}")
                self.exploration_state_ = PlannerType.OBJECT_IDENTIFIED_SCAN
    ##################################################################################################################################################################

    # MAIN LOOP AND EXPLORATION PLANNER ##################################################################################################################################
    def main_loop(self):
        rospy.loginfo("move_base connected")

        # Publisher for the camera detections
        self.image_pub_ = rospy.Publisher("detections_image", Image, queue_size=5)

        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

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

        self.prm_graph_pub_ = rospy.Publisher('prm_graph', Marker, queue_size=10)

        self.lidar_sub_ =  rospy.Subscriber('/scan', LaserScan, self.scan_callback,queue_size=1)

        self.nodes_ = []
        self.potential_node = 0

    def get_pose_2d(self):

        # Lookup the latest transform
        (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = trans[0]
        pose.y = trans[1]

        qw = rot[3];
        qz = rot[2];

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw));

        print("pose: ", pose)

        return pose


    def image_callback(self, image_msg):
        classes=["Alien", "Mineral", "Orb", "Ice", "Muschroom", "Stop Sign"]

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

                    # Draw rectangle and label
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the modified CV2 image back to a ROS Image message
        try:
            processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Publish the processed image
        self.image_pub_.publish(processed_msg)
        rospy.loginfo("Published processed image")


    def planner_move_forwards(self, action_state):
        # Simply move forward by 10m

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
            self.move_base_action_client_.send_goal(action_goal.goal)


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
            self.move_base_action_client_.send_goal(action_goal.goal)



    def planner_return_home(self, action_state):
        # Go to the origin

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
            self.move_base_action_client_.send_goal(action_goal.goal)

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
            self.move_base_action_client_.send_goal(action_goal.goal)

    def planner_random_goal(self, action_state):
        # Go to a random location out of a predefined set

        # Hand picked set of goal locations
        random_goals = [[53.3,40.7],[44.4, 13.3],[2.3, 33.4],[9.9, 37.3],[3.4, 18.5],[6.0, 0.4],[28.3, 11.8],[43.7, 12.8],[38.9,43.0],[47.4,4.7],[31.5,3.2],[36.6,32.5]]

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
            self.move_base_action_client_.send_goal(action_goal.goal)

    # Process lidar and check if a predictive node can be added to the prm
    def scan_callback(self, scan_msg):
        obstructed = 0
        distance_threshold = 6 # How much free space is required to add a node
        distance_set = 4 # How far away from current position node should be
        indices_of_interest=[355, 356, 357, 358, 0, 1, 2, 3, 4] # 10 degree window
        readings_in_range = [scan_msg.ranges[i] for i in indices_of_interest]

        # Check if target range is clear and suitable for a point in the graph
        for i in range(len(readings_in_range)):
            if readings_in_range[i] < distance_threshold:
                obstructed = 1   
        if obstructed == 1:
            self.potential_node = 0
        else:
            pose = self.get_pose_2d()
            front_x = pose.x + distance_set * math.cos(pose.theta)
            front_y = pose.y + distance_set * math.sin(pose.theta)
            self.potential_node = Node(front_x, front_y, 0, 0)

    # Update and connect nodes in the PRM
    def update_prm(self):
        distance_threshold = 4 # How far away nodes should be
        distance = distance_threshold + 1
        node_num = len(self.nodes_)
        closest_node = 0
        
        pose = self.get_pose_2d()

        # If no nodes exist, publish current position
        if node_num == 0:
            self.nodes_.append(Node(pose.x, pose.y, node_num, 0))
            self.publish_prm()
        else :
            # Find closest node to current position
            for node in self.nodes_:
                current_distance = math.sqrt((pose.x - node.x) ** 2 + (pose.y - node.y) ** 2)
                if current_distance < distance:
                    distance = current_distance
                    closest_node = node.idx

            # If the node is further than threshold, publish it
            if distance > distance_threshold:
                self.nodes_.append(Node(pose.x, pose.y, node_num, closest_node))
                self.publish_prm()

            # Alternatively, there may be a potential node in front of the robot
            elif self.potential_node != 0:
                distance = 9999999999
                proximity = 0
                # Check whether this node is suitable based on proximity to existing nodes
                for node in self.nodes_:
                    current_distance = math.sqrt((self.potential_node.x - node.x) ** 2 + (self.potential_node.y - node.y) ** 2)
                    if current_distance < distance:
                        distance = current_distance
                        closest_node = node.idx
                    if distance_threshold > current_distance:
                        proximity = 1
                if proximity == 0:
                    self.nodes_.append(Node(self.potential_node.x, self.potential_node.y, node_num, closest_node))
                    self.publish_prm()

    # Update PRM Visualisation
    def publish_prm(self):
        # Get number of nodes in graph
        node_num = len(self.nodes_) - 1

        # Graph Point publisher
        point_marker = Marker()
        point_marker.header.frame_id = "map"  # Use the appropriate frame
        point_marker.header.stamp = rospy.Time.now()
        point_marker.ns = "points"
        point_marker.id = node_num
        point_marker.type = Marker.SPHERE
        point_marker.action = Marker.ADD

        point_marker.pose.position = Point(self.nodes_[node_num].x, self.nodes_[node_num].y, 0)
        point_marker.pose.orientation.w = 1.0  # No rotation
        point_marker.scale.x = 1  # Sphere radius
        point_marker.scale.y = 1
        point_marker.scale.z = 1
        point_marker.color.r = 1.0
        point_marker.color.g = 0.0
        point_marker.color.b = 1.0
        point_marker.color.a = 1.0
        point_marker.lifetime = rospy.Duration(0)  # Marker will not disappear
        
        self.prm_graph_pub_.publish(point_marker)

        # Graph Line publisher
        if node_num > 0:
            line_marker = Marker()
            line_marker.header.frame_id = "map"  # Use the appropriate frame
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "edges"
            line_marker.id = node_num
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            # Find all nodes that are 6 units or closer and connect them through the graph
            for node in self.nodes_:
                current_distance = math.sqrt((self.nodes_[node_num].x - node.x) ** 2 + (self.nodes_[node_num].y - node.y) ** 2)
                # Find all nodes closer than 6 units and connect them together
                if current_distance < 6:
                    line_marker.points.append(Point(self.nodes_[node_num].x, self.nodes_[node_num].y, 0))
                    line_marker.points.append(Point(node.x, node.y, 0))

            line_marker.pose.orientation.x = 0.0
            line_marker.pose.orientation.y = 0.0
            line_marker.pose.orientation.z = 0.0
            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = 0.25
            line_marker.scale.y = 0.25
            line_marker.scale.z = 0.25
            line_marker.color.r = 0.5
            line_marker.color.g = 0.0
            line_marker.color.b = 0.5
            line_marker.color.a = 1.0 
            line_marker.lifetime = rospy.Duration(0)  # Marker will not disappear
            
            self.prm_graph_pub_.publish(line_marker)

    def main_loop(self):
        # Give everything time to launch
        rospy.sleep(1)

        while not rospy.is_shutdown():
            action_state = self.move_base_action_client_.get_state()
            print("Current State:", self.exploration_state_.name)
            self.exploration_planner(action_state)
            rospy.sleep(0.1)



    def exploration_planner(self, action_state): 
        if self.exploration_state_ == PlannerType.WAITING_FOR_MAP:
            self.handle_waiting_for_map()
            
            
        elif self.exploration_state_ == PlannerType.SELECTING_FRONTIER or self.exploration_state_ == PlannerType.HANDLE_REJECTED_FRONTIER:
            frontier_points = self.find_frontiers()
            self.group_frontiers(frontier_points)
            self.find_min_frontier()
            if not self.chosen_frontier_pose:
                rospy.logwarn('No frontier selected.')
                self.exploration_state_ = PlannerType.EXPLORED_MAP
            else:
                rospy.loginfo('Frontier selected')
            self.exploration_state_ = PlannerType.MOVING_TO_FRONTIER
            
            
        elif self.exploration_state_ == PlannerType.MOVING_TO_FRONTIER:
            self.handle_moving_to_frontier(action_state)
            
            
        elif self.exploration_state_ == PlannerType.HANDLE_TIMEOUT:
            rospy.loginfo("@ handle_timeout")
            self.exploration_state_ = PlannerType.WAITING_FOR_MAP
            
            
        elif self.exploration_state_ == PlannerType.EXPLORED_MAP:
            rospy.loginfo("Exploration completed successfully.")
            rospy.sleep(1.0)
            
            
        elif self.exploration_state_ == PlannerType.OBJECT_IDENTIFIED_SCAN:
            rospy.loginfo("Object identified.")
            self.object_identified_scan()
    ################################################################################################################################################################################


    def handle_waiting_for_map(self):
        while self.occupancy_grid is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(0.5)
        rospy.loginfo('Map acquired')
        self.exploration_state_ = PlannerType.SELECTING_FRONTIER


    # OPTIMUM FRONTIER ALGORITHMS ##########################################################################################################################################################
    def find_frontiers(self):
        frontier_points = []
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                value = self.occupancy_grid[y * self.MAP_WIDTH + x]
                if 0 <= value <= self.INTENSITY_THRESHOLD and self.has_unknown_neighbor(x, y):
                    frontier_points.append((x, y))
        return frontier_points


    def has_unknown_neighbor(self, x, y):
        neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
        for nx, ny in neighbors:
            if 0 <= nx < self.MAP_WIDTH and 0 <= ny < self.MAP_HEIGHT:
                index = ny * self.MAP_WIDTH + nx
                if self.occupancy_grid[index] == -1:
                    return True
        return False


    def group_frontiers(self, frontier_points, eps=1.0, min_samples=2):
        frontier_array = np.array(frontier_points)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(frontier_array)
        labels = db.labels_
        self.frontier_groups = []
        for point, label in zip(frontier_points, labels):
            if label == -1:
                continue
            while len(self.frontier_groups) <= label:
                self.frontier_groups.append([])
            self.frontier_groups[label].append(point)

        self.average_and_size_frontier_points = []
        for points in self.frontier_groups:
            n = len(points)
            if n > self.MIN_CLUSTER_POINTS:
                avg_x = sum(point[0] for point in points) / n
                avg_y = sum(point[1] for point in points) / n
                self.average_and_size_frontier_points.append(((avg_x, avg_y), n))


    # def find_min_frontier(self):
    #     # Find the closest frontier cluster
    #     robot_pose = self.get_pose_2d()
    #     target = []
    #     min_cost = float('inf')
    #     for group in self.average_and_size_frontier_points:
    #         cost = self.group_cost(robot_pose, group)
    #         #print("Cost ", cost)
    #         if cost < min_cost:
    #             min_cost = cost
    #             (x, y), n = group
    #             target = (x, y)

    #     pose_2d = Pose2D()
    #     # Origin Point is 10, 10, and Map Resolution is 0.1
    #     pose_2d.x = target[0] * 0.1 - 10
    #     pose_2d.y = target[1] * 0.1 - 10
    #     pose_2d.theta = random.uniform(0, 2*math.pi)

    #     self.chosen_frontier_pose = pose_2d
    def find_min_frontier(self):
        # Find the closest frontier cluster
        robot_pose = self.get_pose_2d()
        if robot_pose is None:
            rospy.logerr("Could not get robot pose, skipping frontier selection")
            self.chosen_frontier_pose = None
            return

        target = []
        min_cost = float('inf')
        for group in self.average_and_size_frontier_points:
            cost = self.group_cost(robot_pose, group)
            if cost < min_cost:
                min_cost = cost
                (x, y), n = group
                target = (x, y)

        if not target:
            self.chosen_frontier_pose = None
            return

        # Create the Pose2D message properly
        self.chosen_frontier_pose = Pose2D(
            x=target[0] * 0.1 - 10,
            y=target[1] * 0.1 - 10,
            theta=random.uniform(0, 2*math.pi)
        )


    def group_cost(self, current_position, group):
        (avg_x, avg_y), n = group
        distance = math.hypot((current_position.x - avg_x), (current_position.y - avg_y))
        # Distance should be > 1, Length should be < 1
        cost = self.DIST_WEIGHT * (distance**5) - (self.LENGTH_WEIGHT**2) * n

        return cost
    ########################################################################################################################################################################################################

    # MOVING TO FRONTIER ###########################################################################################################################################################################
    def handle_moving_to_frontier(self, action_state):
        
        self.send_goal_Pose(self.chosen_frontier_pose)
        
        try:
            rospy.loginfo('Moving to frontier')
            start_time = rospy.Time.now()
            timeout_duration = rospy.Duration(CaveExplorer.TIME_OUT_MAX)

            while not rospy.is_shutdown():
                if self.exploration_state_ == PlannerType.OBJECT_IDENTIFIED_SCAN:
                    # CANCEL GOAL
                    self.move_base_action_client_.cancel_goal()
                    break
                if rospy.Time.now() - start_time >= timeout_duration:
                    rospy.loginfo("Goal timeout reached")
                    self.move_base_action_client_.cancel_goal()
                    self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                    break

                action_state = self.move_base_action_client_.get_state()
                if action_state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Frontier goal reached")
                    self.exploration_state_ = PlannerType.SELECTING_FRONTIER
                    break
                elif action_state in [actionlib.GoalStatus.PREEMPTED, actionlib.GoalStatus.ABORTED]:
                    rospy.logwarn("Goal aborted or preempted")
                    self.exploration_state_ = PlannerType.HANDLE_REJECTED_FRONTIER
                    break
                rospy.sleep(0.1)
        except Exception as e:
            rospy.logerr(f"Exception while moving to frontier: {e}")


    def object_identified_scan(self):
        # Move to the Location  ###################################################################
        # First Modify so that robot doesn't move all the way to the coordinate, with an offset
        self.offset_coordinates()
        self.send_goal_simple(self.artefact_x_y) 
        rospy.loginfo(f'Moving to Artefact Location, x:{self.artefact_x_y[0]} y:{self.artefact_x_y[1]}')

        # Continuously check the state while moving toward the goal
        action_state = self.move_base_action_client_.get_state()
        
        while action_state != actionlib.GoalStatus.SUCCEEDED:  # 10-second timeout
            # rospy.loginfo(f'Moving to Artefact Location, x:{self.artefact_x_y[0]} y:{self.artefact_x_y[1]}')
            rospy.sleep(0.2)
            
            if action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED, actionlib.GoalStatus.PREEMPTED}:
                rospy.logerr(f"Goal failed with state: {action_state}, RETRYING!!!!!!!!!!!!")
                return
                
            action_state = self.move_base_action_client_.get_state()
        #############################################################################################
        
        # NEXT STATE ################################################################################
        self.exploration_state_ = PlannerType.SELECTING_FRONTIER
        #############################################################################################
        
        
    def offset_coordinates(self):
        # Modify the coordinates to move a specified distance away from the target artifact.
        x, y, theta_target = self.artefact_x_y
        SAFE_DISTANCE = 2.75  # Metres away from the artifact
        
        try:
            # Get current robot position from tf
            (trans, rot) = self.tf_listener_.lookupTransform('/map', '/base_link', rospy.Time(0))
            current_x = trans[0]
            current_y = trans[1]
            
            # Calculate vector from current position to artifact
            dx = x - current_x
            dy = y - current_y
            
            # Calculate angle to artifact
            angle_to_artifact = math.atan2(dy, dx)
            
            # Calculate offset position
            offset_x = x - SAFE_DISTANCE * math.cos(angle_to_artifact)
            offset_y = y - SAFE_DISTANCE * math.sin(angle_to_artifact)
            
            # Update artifact coordinates with offset position and original orientation
            self.artefact_x_y = (offset_x, offset_y, theta_target)
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Could not get transform from map to base_link: {str(e)}")
            # If we can't get the transform, just offset in the direction of theta_target
            offset_x = x - SAFE_DISTANCE * math.cos(theta_target)
            offset_y = y - SAFE_DISTANCE * math.sin(theta_target)
            self.artefact_x_y = (offset_x, offset_y, theta_target)
    ################################################################################################################################################
         
         
    # SEND GOALS & GET POSE ########################################################################################################################
    # def get_pose_2d(self):
    #     # Lookup the latest transform
    #     (trans,rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))

    #     # Return a Pose2D message
    #     pose = Pose2D()
    #     pose.x = trans[0]
    #     pose.y = trans[1]
    #     qw = rot[3]
    #     qz = rot[2]

    #     if qz >= 0.:
    #         pose.theta = wrap_angle(2. * math.acos(qw))
    #     else: 
    #         pose.theta = wrap_angle(-2. * math.acos(qw))

    #     return pose
    def get_pose_2d(self):
        # Lookup the latest transform
        try:
            (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
            
            # Create and return a Pose2D message
            pose = Pose2D()
            # Use the proper message constructor method
            return Pose2D(
                x=trans[0],
                y=trans[1],
                theta=wrap_angle(2. * math.acos(rot[3]) if rot[2] >= 0. else -2. * math.acos(rot[3]))
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform: {e}")
            return None

         
    
    def send_goal(self, frontier):
        
        map_resolution = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        # send first selected frontier to move_base and remove it from, till all the frontiers are sent to move_base
        x, y = frontier
        # Send a goal to move_base to explore the selected frontier
        pose_2d = Pose2D
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
        
        
    # def send_goal_simple(self, coordinate):
    #     pose_2d = Pose2D
    #     pose_2d.x = coordinate[0]
    #     pose_2d.y = coordinate[1]
    #     pose_2d.theta = coordinate[2]
    #     # print(f'x:{pose_2d.x} , y:{pose_2d.y}')

    #     # Send a goal to "move_base" with "self.move_base_action_client_"
    #     action_goal = MoveBaseActionGoal()

    #     action_goal.goal.target_pose.header.frame_id = "map"
    #     action_goal.goal_id = self.goal_counter_
    #     self.goal_counter_ = self.goal_counter_ + 1
    #     action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

    #     # sending the goal to move base
    #     self.move_base_action_client_.send_goal(action_goal.goal)
    def send_goal_simple(self, coordinate):
        # Create the Pose2D message properly
        pose_2d = Pose2D(
            x=coordinate[0],
            y=coordinate[1],
            theta=coordinate[2]
        )

        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id.id = str(self.goal_counter_)
        self.goal_counter_ += 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)

    def send_goal_Pose(self, target_pose):
        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()

        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ = self.goal_counter_ + 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(target_pose)

        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)
        
    def send_goal(self, frontier):
        x, y = frontier
        # Create the Pose2D message properly
        pose_2d = Pose2D(
            x=x * self.current_map_.info.resolution + self.current_map_.info.origin.position.x,
            y=y * self.current_map_.info.resolution + self.current_map_.info.origin.position.y,
            theta=math.pi/2
        )

        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()
        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id.id = str(self.goal_counter_)
        self.goal_counter_ += 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)
############################################################################################################################################
                
                
if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('cave_explorer', anonymous=True)
    # Create the cave explorer
    cave_explorer = CaveExplorer()
    # Loop forever while processing callbacks
    cave_explorer.main_loop()