#!/usr/bin/env python3

import rospy
import roslib
import math
import cv2 # OpenCV2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from nav_msgs.srv import GetMap
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
import tf
from std_srvs.srv import Empty
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import random
import copy
from threading import Lock
from enum import Enum



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
    OBJECT_IDENTIFIED = 8
    SCANNING_OBJECT = 9
    EXPLORED_MAP = 10

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
        self.exploration_done_ = False
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        self.visited_frontiers = set()
        
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
        self.image_detections_pub_ = rospy.Publisher('detections_image', Image, queue_size=1)

        # Read in computer vision model (simple starting point)
        self.computer_vision_model_filename_ = rospy.get_param("~computer_vision_model_filename")
        self.computer_vision_model_ = cv2.CascadeClassifier(self.computer_vision_model_filename_)

        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)

        # Subscribe to the map topic
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.current_map_ = None
    
        #Subscriber of the pose
        self.odom_sub_ = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.odom_ = None

        # map service
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

        # print("pose: ", pose)

        return pose


    def image_callback(self, image_msg):
        pass


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
        

    
    def exploration_planner(self, action_state):
        if action_state != actionlib.GoalStatus.ACTIVE:
            print('Exploration planner ............')
            if self.exploration_state_ == ExplorationsState.WAITING_FOR_MAP:
                self.handle_waiting_for_map()

            elif self.exploration_state_ == ExplorationsState.IDENTIFYING_FRONTIERS:
                self.handle_identifying_frontiers()

            elif self.exploration_state_ == ExplorationsState.SELECTING_FRONTIER:
                self.handle_selecting_frontier()

            elif self.exploration_state_ == ExplorationsState.MOVING_TO_FRONTIER:
                self.handle_moving_to_frontier(action_state)

            elif self.exploration_state_ == ExplorationsState.HANDLE_REJECTED_FRONTIER:
                self.handle_selecting_frontier()
                
            elif self.exploration_state_ == ExplorationsState.HANDLE_TIMEOUT:
                self.handle_timeout()

            elif self.exploration_state_ == ExplorationsState.EXPLORED_MAP:
                rospy.loginfo("Exploration completed successfully.")

    def handle_waiting_for_map(self):
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
        print ( 'identifying frontiers')
        frontiers = self.identify_frontiers(self.current_map_)
        if not frontiers:
            rospy.loginfo('No frontiers found!')
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
        else:
            self.exploration_state_ = ExplorationsState.SELECTING_FRONTIER

    def handle_selecting_frontier(self):
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
        for frontier in self.selected_frontier:
            if tuple(frontier) in self.visited_frontiers:
                continue # Skip the frontier if it has already been visited
        
            self.move_to_frontier(frontier)
            rospy.loginfo('Moving to frontier')
            start_time = rospy.Time.now()
            removed_frontier = self.selected_frontier.pop(0)
            self.visited_frontiers = self.visited_frontiers.union({tuple(removed_frontier)})

            action_state = self.move_base_action_client_.get_state()
            print ( ' check action state')
            if action_state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Successfully reached the frontier!")
                    self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        
            elif action_state in {actionlib.GoalStatus.REJECTED, actionlib.GoalStatus.ABORTED}:
                    rospy.loginfo("Goal rejected or aborted.")
                    # self.visited_frontiers.add(tuple(frontier))
                    frontier = None
                    self.exploration_state_ = ExplorationsState.HANDLE_REJECTED_FRONTIER
            elif rospy.Time.now() - start_time > rospy.Duration(10):  # Timeout after 10 seconds
                rospy.loginfo("Timeout reached.")
                self.exploration_state_ = ExplorationsState.HANDLE_TIMEOUT
                
                rospy.sleep(0.1)
                                        
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
        if frontiers:
            rospy.loginfo(f'Frontiers found: {len(frontiers)}')
            frontiers_merged= self.merge_close_frontiers_to_one(frontiers)
            rospy.loginfo(f'Frontiers merged: {len(frontiers_merged)}')
            frontiers_merged.sort(key=self.compute_distance)
            rospy.loginfo(f'Frontiers sorted: {len(frontiers_merged)}')
            # print ( 'frontiers sorted')
            return frontiers_merged
        return None
    
    def merge_close_frontiers_to_one(self, frontiers):
        merged_frontiers = []
        threshold_distance = 30  # Distance threshold to consider frontiers as close

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
        if self.odom_:
            map_res = self.current_map_.info.resolution
            map_origin = self.current_map_.info.origin.position
            x, y = frontier
            frontier_x = x * map_res + map_origin.x
            frontier_y = y * map_res + map_origin.y
            distance = np.sqrt((self.current_position.x - frontier_x) ** 2 + (self.current_position.y - frontier_y) ** 2)
            return distance
        return float('inf')
        
    
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
            
    def handle_timeout(self):
        rospy.loginfo("Timeout reached.")
        self.current_map_ = None
        self.selected_frontier = None
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        
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




