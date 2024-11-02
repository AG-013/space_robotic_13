#!/usr/bin/env python3

# Math Modules
import math
import numpy as np
import random
from sklearn.cluster import DBSCAN

# ROS Modules
import tf
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose2D

from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
import actionlib
import rospy
from std_srvs.srv import Trigger

from helper_functions import *
from enums import *


class CaveExplorer:
    
    TIME_OUT_MAX = 27.5
    MAP_WIDTH = 896
    MAP_HEIGHT = 896
    MIN_CLUSTER_POINTS = 40
    INTENSITY_THRESHOLD = 15
    LENGTH_WEIGHT = 1.4
    DIST_WEIGHT = 2
    SAFE_DISTANCE = 2.75  # Metres away from the artifact
    GOAL_THRESHOLD = 1.0
    MAP_RESOLUTION = 0.1
    MAP_ORIGIN_X = 10
    MAP_ORIGIN_Y = 10
    
    def __init__(self):
        rospy.init_node('cave_explorer', anonymous=True)
        
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

    def timer_artifact_callback(self, event):
        coords = self.get_artifact_location()
        if coords is not None:
            if coords not in self.artefacts_list:
                self.artefacts_list.append(coords)
                # self.artefact_x_y = coords
                self.artefact_x_y = Pose2D(coords[0],
                                           coords[1],
                                           coords[2])
                # rospy.loginfo(f"NEW: Found artifact at X: {self.artefact_x_y[0]:.2f}, Y: {self.artefact_x_y[1]:.2f}, Theta: {self.artefact_x_y[2]:.2f}")
                self.exploration_state_ = PlannerType.OBJECT_IDENTIFIED_SCAN
                
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
    ##################################################################################################################################################################

    # MAIN LOOP AND EXPLORATION PLANNER ##################################################################################################################################
    def main_loop(self):
        while not rospy.is_shutdown():
            action_state = self.move_base_action_client_.get_state()
            print("Current State:", self.exploration_state_.name)
            self.exploration_planner(action_state)
            rospy.sleep(0.1)


    def exploration_planner(self, action_state): 
        if self.exploration_state_ == PlannerType.WAITING_FOR_MAP:
            self.handle_waiting_for_map()
            
        elif self.exploration_state_ == PlannerType.SELECTING_FRONTIER or self.exploration_state_ == PlannerType.HANDLE_REJECTED_FRONTIER:
            self.handle_frontier_finder()            
            
        elif self.exploration_state_ == PlannerType.MOVING_TO_FRONTIER:
            self.handle_moving_to_frontier(action_state)
            
        elif self.exploration_state_ == PlannerType.HANDLE_TIMEOUT:
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
    def handle_frontier_finder(self):
        frontier_points = self.find_frontiers()
        self.group_frontiers(frontier_points)
        self.find_min_frontier()
        if not self.chosen_frontier_pose:
            rospy.logwarn('No frontier selected.')
            self.exploration_state_ = PlannerType.EXPLORED_MAP
            return
        else:
            rospy.loginfo('Frontier selected')
            self.exploration_state_ = PlannerType.MOVING_TO_FRONTIER
    
    def find_frontiers(self):
        frontier_points = []
        for y in range(CaveExplorer.MAP_HEIGHT):
            for x in range(CaveExplorer.MAP_WIDTH):
                value = self.occupancy_grid[y * CaveExplorer.MAP_WIDTH + x]
                if 0 <= value <= CaveExplorer.INTENSITY_THRESHOLD and self.has_unknown_neighbor(x, y):
                    frontier_points.append((x, y))
        return frontier_points

    def has_unknown_neighbor(self, x, y):
        neighbors = [(x - 1, y),
                     (x + 1, y),
                     (x, y - 1),
                     (x, y + 1)]
        
        for nx, ny in neighbors:
            if 0 <= nx < CaveExplorer.MAP_WIDTH and 0 <= ny < CaveExplorer.MAP_HEIGHT:
                index = ny * CaveExplorer.MAP_WIDTH + nx
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
            if n > CaveExplorer.MIN_CLUSTER_POINTS:
                avg_x = sum(point[0] for point in points) / n
                avg_y = sum(point[1] for point in points) / n
                self.average_and_size_frontier_points.append(((avg_x, avg_y), n))


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
        self.chosen_frontier_pose = Pose2D( x=target[0] * CaveExplorer.MAP_RESOLUTION - CaveExplorer.MAP_ORIGIN_X,
                                            y=target[1] * CaveExplorer.MAP_RESOLUTION - CaveExplorer.MAP_ORIGIN_Y,
                                            theta=random.uniform(0, 2*math.pi))


    # def group_cost(self, current_position, group):
    #     (avg_x, avg_y), n = group
    #     distance = math.hypot((current_position.x - avg_x), (current_position.y - avg_y))
    #     # Distance should be > 1, Length should be < 1
    #     # cost = CaveExplorer.DIST_WEIGHT * (distance**4) - (CaveExplorer.LENGTH_WEIGHT) * n
    #     cost = CaveExplorer.DIST_WEIGHT * (distance**4) - (CaveExplorer.LENGTH_WEIGHT) * (n ** 2)
    #     return cost
    
    def group_cost(self, current_position, group):
        (avg_x, avg_y), n = group
        # Convert grid coordinates to map frame
        frontier_x = avg_x * CaveExplorer.MAP_RESOLUTION - CaveExplorer.MAP_ORIGIN_X
        frontier_y = avg_y * CaveExplorer.MAP_RESOLUTION - CaveExplorer.MAP_ORIGIN_Y
        
        # Now both points are in meters in the map frame
        distance = math.hypot(  (current_position.x - frontier_x), 
                                (current_position.y - frontier_y))
        
        cost = ((CaveExplorer.DIST_WEIGHT * distance)**4) - (CaveExplorer.LENGTH_WEIGHT) * (n ** 2)
        
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
                action_state = self.move_base_action_client_.get_state()
                
                if self.exploration_state_ != PlannerType.MOVING_TO_FRONTIER:
                    self.move_base_action_client_.cancel_goal()
                    break

                elif rospy.Time.now() - start_time >= timeout_duration:
                    rospy.loginfo("Goal timeout reached")
                    self.move_base_action_client_.cancel_goal()
                    self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                    break

                elif action_state == actionlib.GoalStatus.SUCCEEDED:
                    rospy.loginfo("Frontier goal reached")
                    self.exploration_state_ = PlannerType.SELECTING_FRONTIER
                    break

                elif action_state in [actionlib.GoalStatus.PREEMPTED, actionlib.GoalStatus.ABORTED]:
                    rospy.logwarn("Goal aborted or preempted")
                    self.exploration_state_ = PlannerType.HANDLE_REJECTED_FRONTIER
                    break

                elif self.dist_to_goal() < CaveExplorer.GOAL_THRESHOLD:
                    rospy.logerr('OPTIMISING AS I\'M CLOSE ENOUGH')
                    self.move_base_action_client_.cancel_goal()
                    self.exploration_state_ = PlannerType.SELECTING_FRONTIER
                    break
                
                rospy.sleep(0.1)
                
        except Exception as e:
            rospy.logerr(f"Exception while moving to frontier: {e}")


    def dist_to_goal(self):
        # find distance between current location and goal
        cur_pose = self.get_pose_2d()
        return compute_distance_between_points( (self.chosen_frontier_pose.x, self.chosen_frontier_pose.y),
                                                (cur_pose.x, cur_pose.y))
        
        
    def object_identified_scan(self):
        # Move to the Location  ###################################################################
        # First Modify so that robot doesn't move all the way to the coordinate, with an offset
        self.offset_coordinates()
        
        self.send_goal_Pose(self.artefact_x_y) 
        # rospy.loginfo(f'Moving to Artefact Location, x:{self.artefact_x_y[0]} y:{self.artefact_x_y[1]}')

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
        

    # Modify the coordinates to move a specified distance away from the target artifact.        
    def offset_coordinates(self):
        x = self.artefact_x_y.x
        y = self.artefact_x_y.y
        theta_target = self.artefact_x_y.theta
        
        # Calculate offset position
        offset_x = x - CaveExplorer.SAFE_DISTANCE * math.cos(theta_target)
        offset_y = y - CaveExplorer.SAFE_DISTANCE * math.sin(theta_target)
        
        # Update artifact coordinates with offset position and original orientation
        self.artefact_x_y = Pose2D( x=offset_x,
                                    y=offset_y,
                                    theta=theta_target)
    ################################################################################################################################################
         
         
    # SEND GOALS & GET POSE ########################################################################################################################
    def get_pose_2d(self):
        # Lookup the latest transform
        try:
            (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
            
            # Use the proper message constructor method
            return Pose2D(  x=trans[0],
                            y=trans[1],
                            theta=wrap_angle(2. * math.acos(rot[3]) if rot[2] >= 0. else -2. * math.acos(rot[3])) )
            
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr(f"Failed to get transform: {e}")
            return None
         
    
    def send_goal(self, frontier):
        
        map_resolution = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        # send first selected frontier to move_base and remove it from, till all the frontiers are sent to move_base
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
        
        
    def send_goal_Pose(self, pose):
        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()

        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ = self.goal_counter_ + 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose)

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