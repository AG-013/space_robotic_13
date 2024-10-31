import math
import numpy as np
import random
from move_base_msgs.msg import MoveBaseAction, MoveBaseActionGoal
from geometry_msgs.msg import Pose2D
import actionlib
from sklearn.cluster import DBSCAN


MAP_WIDTH = 896
MAP_HEIGHT = 896
MIN_CLUSTER_POINTS = 50


class FrontierPlanner:
    def __init__(self):
        self.occupancy_grid = None
        self.threshold = 10
        self.frontier_groups = []
        self.average_frontier_points = []
        self.goal_counter_ = 0
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    def map_callback(self, msg):
        self.occupancy_grid = msg.data
        frontier_points = self.find_frontiers()

    def find_frontiers(self):
        frontier_points = []
        # Iterate over the occupancy grid
        for col_n in range(MAP_HEIGHT):
            for row_n in range(MAP_WIDTH):
                value = self.occupancy_grid[col_n * MAP_WIDTH + row_n]

                # Known point (0-100) with unknown neighbors is a frontier
                if 0 <= value <= self.threshold and self.has_unknown_neighbor(row_n, col_n):
                    frontier_points.append((row_n, col_n))

        return frontier_points

    def has_unknown_neighbor(self, x, y):
        neighbors = [(x - 1, y),    # Left
                     (x + 1, y),    # Right
                     (x, y - 1),    # Down
                     (x, y + 1)]    # Up
        
        for nx, ny in neighbors:
            if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
                index = ny * MAP_WIDTH + nx
                if self.occupancy_grid[index] == -1:  # Unknown
                    return True
                
        return False

    def group_frontiers(self, frontier_points, eps=1.0, min_samples=2):
        # Convert frontier points to a NumPy array for DBSCAN
        frontier_array = np.array(frontier_points)
        # Apply DBSCAN clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(frontier_array)
        # Get the labels assigned by DBSCAN (-1 means noise)
        labels = db.labels_

        # Initialize a list to hold clusters
        self.frontier_groups = []
        # Group points by their cluster label
        for point, label in zip(frontier_points, labels):
            if label == -1:
                # Skip noise points
                continue
            # Ensure that the cluster index exists in the clusters list
            while len(self.frontier_groups) <= label:
                self.frontier_groups.append([])  # Add a new empty cluster if it doesn't exist
            self.frontier_groups[label].append(point)  # Append point to the correct cluster

        # Calculate average points for each group
        average_points = []
        for group in self.frontier_groups:
            n = len(group)
            if n > MIN_CLUSTER_POINTS:  # Checks if the group has more than 50 points
                # Calculate the average x and y
                average_x = sum(point[0] for point in group) / n
                average_y = sum(point[1] for point in group) / n
                # Store the average point as a tuple in the average_points list
                average_points.append((average_x, average_y))
                
        self.average_frontier_points = average_points

    def group_cost(self, current_position, group, weight=1):
        average_x = sum(point[0] for point in group) / len(group)
        average_y = sum(point[1] for point in group) / len(group)

        distance = math.hypot(current_position.x - average_x, current_position.y - average_y)
        cost = (distance * 0.2) - weight * len(group)

        return cost, average_x, average_y

    def planner_move_to_frontier(self, action_state):
        # Find the closest frontier cluster
        if action_state != actionlib.GoalStatus.ACTIVE and self.average_frontier_points:
            robot_pose = self.get_pose_2d()
            target = []
            min_cost = float('inf')
            for group in self.frontier_groups:
                if len(group) > 40:
                    cost, x, y = self.group_cost(robot_pose, group)
                    if cost < min_cost:
                        min_cost = cost
                        target = (x, y)

            pose_2d = Pose2D()
            pose_2d.x = target[0] * 0.1 - 10
            pose_2d.y = target[1] * 0.1 - 10
            pose_2d.theta = random.uniform(0, 2 * math.pi)

            # Send a goal to "move_base" with "self.move_base_action_client_"
            action_goal = MoveBaseActionGoal()
            action_goal.goal.target_pose.header.frame_id = "map"
            action_goal.goal_id.stamp = self.goal_counter_
            self.goal_counter_ += 1
            action_goal.goal.target_pose.pose = self.pose2d_to_pose(pose_2d)

            self.move_base_action_client_.send_goal(action_goal.goal)