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
from geometry_msgs.msg import Twist, Pose2D, Pose, Point
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
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


def compute_distance_between_points(point1, point2):
    return np.hypot((point1[0] - point2[0]), (point1[1] - point2[1]))
    
    
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

        # Wait for the transform from map to base_link
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        # Subscribers
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

        # Publishers
        self.cmd_vel_pub_ = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.image_pub_ = rospy.Publisher("/detections_image", Image, queue_size=5)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")

        # Other initialization code...
        self.stop_threads = threading.Event()  # Use an event to signal threads to stop

        # Start the image processing thread
        self.image_thread = threading.Thread(target=self.process_image_continuously, daemon=True)
        self.image_thread.start()

        # Start the exploration thread with a slightly longer sleep interval
        self.exploration_thread = threading.Thread(target=self.exploration_loop, daemon=True)
        self.exploration_thread.start()


    # Callback functions with locks #######################################################
    def image_callback(self, image_msg):
        with self.image_lock:
            self.image_data = image_msg

    def map_callback(self, map_msg):
        with self.map_lock:
            if self.current_map_ is None or self.current_map_.header.stamp != map_msg.header.stamp:
                rospy.loginfo("Map updated")
                self.current_map_ = map_msg

    # # Image processing thread ############################################################
    def process_image_continuously(self):
        # Continuously processes images with a minimal sleep interval for higher priority.
        image_rate = rospy.Rate(20)  # 20 Hz for image processing (adjust as needed)
        try:
            while not rospy.is_shutdown() and not self.stop_threads.is_set():
                with self.image_lock:
                    if self.image_data:
                        self.process_image(self.image_data)
                image_rate.sleep()  # Use rate control for consistent processing frequency
        except rospy.ROSInterruptException:
            rospy.loginfo("Image processing thread shutting down.")

    def exploration_loop(self):
        # Exploration loop with a longer sleep interval to reduce CPU contention.
        exploration_rate = rospy.Rate(0.5)  # 0.5 Hz for exploration (once every 2 seconds)
        try:
            while not rospy.is_shutdown() and not self.stop_threads.is_set():
                self.main_loop()
                exploration_rate.sleep()  # Run exploration at a much lower frequency
        except rospy.ROSInterruptException:
            rospy.loginfo("Exploration thread shutting down.")
            
    def stop_all_threads(self):
        # Method to stop all threads gracefully.
        self.stop_threads.set()
        
    def __del__(self):
        # Ensure all threads are stopped upon object deletion.
        self.stop_all_threads()

    def process_image(self, image_msg):
        # Define the classes for YOLO detection
        classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]

        # Convert the ROS image message to a CV2 image
        try:
            cv_image = self.cv_bridge_.imgmsg_to_cv2(image_msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Process the image using YOLO with reduced image size for efficiency
        results = self.model_(cv_image, device=self.device_, imgsz=(480, 384))

        detected_objects = []  # To store details of detected objects

        # Draw bounding boxes on the image
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0].item()  # Confidence score

                # Only process boxes with confidence above the threshold
                confidence_threshold = 0.65
                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls.item())  # Get the class ID from the tensor
                    label = f'{classes[class_id]} {confidence:.2f}'  # Class name and confidence

                    # Calculate center point of the bounding box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Get the 3D coordinates based on the center point of the bounding box
                    art_xyz = self.get_posed_3d(center_x, center_y)

                    if art_xyz is not None:
                        # Store detected object details for efficient bounding box drawing
                        detected_objects.append((x1, y1, x2, y2, label, center_x, center_y, art_xyz, class_id))

                        # Check if detected object is a "Mineral" or "Mushroom"
                        if class_id in (1, 4):
                            artefact_list = self.mineral_artefacts if class_id == 1 else self.mushroom_artefacts

                            # Check if the artifact already exists within a threshold distance
                            if not self.is_existing_artefact(artefact_list, art_xyz, threshold=7):
                                artefact_list.append(art_xyz)
                                # You can create a thread here to go to the artifact if needed
                                # threading.Thread(target=self.go_to_artifact, args=(art_xyz,)).start()
                    else:
                        # Add objects without 3D coordinates with a warning color
                        detected_objects.append((x1, y1, x2, y2, label, center_x, center_y, None, class_id))

        # Draw all bounding boxes and labels on the image
        for x1, y1, x2, y2, label, center_x, center_y, art_xyz, class_id in detected_objects:
            color = (0, 255, 0) if art_xyz is not None else (0, 0, 255)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

        # Convert the modified CV2 image back to a ROS Image message
        processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")

        # Publish the processed image
        self.image_pub_.publish(processed_msg)

    def is_existing_artefact(self, artefact_list, new_artefact, threshold=7):
        """
        Check if an artefact already exists within a minimum distance threshold.
        """
        for artefact in artefact_list:
            if math.hypot(artefact[0] - new_artefact[0], artefact[1] - new_artefact[1]) <= threshold:
                return True
        return False
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
        
        theta = self.wrap_angle(2.0 * math.acos(qw) if qz >= 0 else -2.0 * math.acos(qw))
        
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


    # Exploration planning thread #########################################################
    def exploration_loop(self):
        while not rospy.is_shutdown():
            self.main_loop()
            rospy.sleep(0.5)

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
        self.cmd_vel_pub_.publish(twist)
        
        # identify object coordinates and move towards it align so that the object is in the center of the camera move towards the object and 
        
        # Return to the exploration state
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        

    def handle_waiting_for_map(self):
        print( 'waiting for map.....')
        while self.current_map_ is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(1.0)  # Wait for the map to be received
            continue  # Keep waiting until the map is received
        # load the current map
        current_map = self.current_map_
        print ( ' map acquired')  
        self.exploration_state_ = ExplorationsState.IDENTIFYING_FRONTIERS
        print ('state changed to identifying frontiers')
    

    def handle_identifying_frontiers(self):
        print ('identifying frontiers,,,,,')
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
                if compute_distance_between_points(random_frontier, frontier) < threshold_distance:
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
        # This method implements the main exploration state machine
        with self.map_lock:
            action_state = self.move_base_action_client_.get_state()
            rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
            rospy.loginfo('action_state number:' + str(action_state))

            if (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.SUCCEEDED):
                rospy.loginfo("Successfully explored!")
                self.exploration_done_ = True
            elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE):
                rospy.loginfo("Exploration preempted!")
                self.exploration_done_ = False
                action_state = actionlib.GoalStatus.PREEMPTING

            # Select the next planner to execute
            if not self.exploration_done_:
                self.planner_type_ = PlannerType.EXPLORATION

            # Execute the planner by calling the relevant method
            rospy.loginfo("Calling planner: " + self.planner_type_.name)
            if self.planner_type_ == PlannerType.EXPLORATION:
                self.exploration_planner(action_state)


if __name__ == '__main__':
    rospy.init_node('cave_explorer')

    # Create the cave explorer instance
    cave_explorer = CaveExplorer()

    # Keep the node running
    rospy.spin()