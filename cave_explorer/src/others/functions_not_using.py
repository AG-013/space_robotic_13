
# NOT USING???

       
# def index_to_position(self, index):
#     # Convert the index of the grid cell to a (x, y) position in the map
#     width = self.current_map_.info.width
#     resolution = self.current_map_.info.resolution
#     origin_x = self.current_map_.info.origin.position.x
#     origin_y = self.current_map_.info.origin.position.y
    
#     x = (index % width) * resolution + origin_x
#     y = (index // width) * resolution + origin_y

    
#     return Point(x, y, 0)  # Return as a Point object



# def is_valid_frontier(self, frontier_point):
#     # Check if the frontier point is far enough from all visited frontiers.
#     min_distance = 0 # Define a minimum distance threshold
    
#     for visited_point in self.visited_frontiers:
#         # Calculate Euclidean distance between the frontier and visited point
#         distance = np.hypot((frontier_point[0] - visited_point[0]), (frontier_point[1] - visited_point[1]))
#         if distance < min_distance:
#             return False  # The frontier is too close to a visited point
        
#     return True



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


        
# def compute_distance(self, frontier):
#     map_res = self.current_map_.info.resolution
#     map_origin = self.current_map_.info.origin.position
#     x, y = frontier
#     frontier_x = x * map_res + map_origin.x
#     frontier_y = y * map_res + map_origin.y
#     distance = np.hypot((frontier_x - self.current_position.x ), (frontier_y - self.current_position.y ))
#     return distance


# def object_identified_scan(self):
#     # Stop the robot
#     twist = Twist()
#     self.cmd_vel_pub_.publish(twist)
#     self.planner_type_ = PlannerType.WAITING_FOR_MAP




# OLD MAIN

# def main_loop(self):

#     while not rospy.is_shutdown():

#         # Get the current status - possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
#         action_state = self.move_base_action_client_.get_state()
#         rospy.loginfo('action state: ' + self.move_base_action_client_.get_goal_status_text())
#         rospy.loginfo('action_state number:' + str(action_state))
        
#         if (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.SUCCEEDED):
#             print("Successfully explored!")
#             self.exploration_done_ = True
#         elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE):
#             print("Exploration preempted!")
#             self.exploration_done_ = False
#             action_state = actionlib.GoalStatus.PREEMPTING
#         # elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE) and (self.exploration_planner == OBJECT_IDENTIFIED_SCAN)
#         #     print (' moving to object  ')
#         #     self.exploration_done_= False
#             ## put in change state and call object scan 

#         # Select the next planner to execute - Update this logic as you see fit!
#         if not self.exploration_done_:
#             self.planner_type_ = PlannerType.EXPLORATION

#         # Execute the planner by calling the relevant method - methods send a goal to "move_base" with "self.move_base_action_client_"
#         print("Calling planner:", self.planner_type_.name)
#         if self.planner_type_ == PlannerType.EXPLORATION: 
#             self.exploration_planner(action_state)

#         # Delay so the loop doesn't run too fast
#         rospy.sleep(0.5)