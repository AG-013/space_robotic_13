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
INTENSITY_THRESHOLD = 10
LENGTH_WEIGHT = 1
DIST_WEIGHT = 1

self.occupancy_grid = None


def map_callback(self, msg):
    self.occupancy_grid = msg.data
    frontier_points = self.find_frontiers(10)


# GROUPING POINTS ##############################################################################
def find_frontiers(self):
    frontier_points = []
    # Iterate over the occupancy grid
    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            value = self.occupancy_grid[y * MAP_WIDTH + x]

            # Known point (0-100) with unknown neighbors is a frontier
            if 0 <= value <= INTENSITY_THRESHOLD and has_unknown_neighbor(x, y):
                frontier_points.append((x, y))

    return frontier_points
# Check all neighbours of the cell to see if its unknown - not check diagonals atm - 
# If at least one of the neighbours is unknown, return true
def has_unknown_neighbor(x, y):
    neighbors = [   (x - 1, y),     # Left
                    (x + 1, y),     # Right
                    (x, y - 1),     # Down
                    (x, y + 1) ]     # Up
    
    for nx, ny in neighbors:
        if 0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT:
            index = ny * MAP_WIDTH + nx
            if self.occupancy_grid[index] == -1:  # Unknown
                return True
            
    return False
#####################################################################################################


# clusters using DBSCAN.
 # :param frontier_points: List of (x, y) frontier coordinates
 # :param eps: Maximum distance between two points to be considered in the same cluster
 # :param min_samples: Minimum number of points required to form a cluster
 # :return: List of clusters, where each cluster is a list of frontier points
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
    self.average_and_size_frontier_points = []
    for points in enumerate(self.frontier_groups):
        n = len(points)
        if n > MIN_CLUSTER_POINTS:  # Checks if the group has more than 50 points
            # Calculate the average x and y
            avg_x = sum(point[0] for point in points) / n
            avg_y = sum(point[1] for point in points) / n
            # Store the average point as a tuple in the average_points list
            self.average_and_size_frontier_points.append(((avg_x, avg_y), n))
            

# Evaluates the best frontier to move towards, and sends that goal to the action server to move the robot to that location
# Frontiers are evalulated based on their average position, and their distance from the current position.
def planner_move_to_frontier(self, action_state):
    # Find the closest frontier cluster
    if action_state != actionlib.GoalStatus.ACTIVE and self.average_and_size_frontier_points:

        robot_pose = self.get_pose_2d()
        target = []
        min_cost = float('inf')
        for group in self.average_and_size_frontier_points:
            cost, x, y = self.group_cost(robot_pose, group)
            #print("Cost ", cost)
            if cost < min_cost:
                min_cost = cost
                target = (x, y)

        pose_2d = Pose2D()
        # Origin Point is 10, 10, and Map Resolution is 0.1
        pose_2d.x = target[0] * 0.1 - 10
        pose_2d.y = target[1] * 0.1 - 10
        pose_2d.theta = random.uniform(0, 2*math.pi)

        self.send_goal_Pose(pose_2d)
        
        
# Returns the cost of a group in terms of it's cluster size and distance. Lower costs are preferable 
# PASS THE FRICKEN AVERAGES
# MAYBE: # Cost = Distance^2 + weight*(cluster size)
def group_cost(self, current_position, group):
    (avg_x, avg_y), n = group
    distance = math.hypot((current_position.x - avg_x), (current_position.y - avg_y))
    # Distance should be > 1, Length should be < 1
    cost = DIST_WEIGHT * distance + LENGTH_WEIGHT * n

    return cost