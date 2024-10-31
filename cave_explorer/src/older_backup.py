



class CaveExplorer:
    
    TIME_OUT_MAX = 27.5
    
    def __init__(self):
        rospy.init_node('cave_explorer', anonymous=True)  # Initialize the ROS node

        # VARIABLES FOR FRONTIER ############################################################################
        self.MAP_WIDTH = 896
        self.MAP_HEIGHT = 896
        self.MIN_CLUSTER_POINTS = 50
        self.INTENSITY_THRESHOLD = 10
        self.LENGTH_WEIGHT = 1
        self.DIST_WEIGHT = 1

        self.occupancy_grid = None
        ####################################################################################################

        self.current_map_ = None

        # Variables/Flags for planning
        self.goal_counter_ = 0  # Unique ID for each goal sent to move_base
        self.exploration_done_ = False
        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
        self.visited_frontiers = set()

        # Wait for the transform from map to base_link
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
            rospy.sleep(0.1)
            rospy.logwarn("Waiting for transform... Have you launched a SLAM node?")

        # Subscribers
        self.map_sub_ = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)

        # Action client for move_base
        self.move_base_action_client_ = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("Waiting for move_base action...")
        self.move_base_action_client_.wait_for_server()
        rospy.loginfo("move_base connected")        
        
        # Service client for getting artifact location
        rospy.wait_for_service('get_artifact_location')
        self.artifact_location_service_client = rospy.ServiceProxy('get_artifact_location', Trigger)
        # Call the service and print the coordinates
        self.get_artifact_location()
        self.artifact_check_timer = rospy.Timer(rospy.Duration(2.0), self.timer_artifact_callback)  # Check every 1 second
        self.artefact_x_y = None
        self.artefacts_list = []


    # Callbacks & Services ################################################################################################
    def map_callback(self, msg):
        self.occupancy_grid = msg.data
        rospy.sleep(0.2)
        
    
    def get_artifact_location(self):
        """Call the artifact location service and print the coordinates."""
        try:
            if self.exploration_state_ != PlannerType.OBJECT_IDENTIFIED_SCAN:
                response = self.artifact_location_service_client()
                
                if response.success:
                    try:
                        x, y, z= map(float, response.message.split(','))
                        # rospy.loginfo(f"Possible new artifact at X: {x:.2f}, Y: {y:.2f}")
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
        """Callback for the timer to regularly check for artifacts"""
        # rospy.loginfo("Checking for artifacts...")
        coords = self.get_artifact_location()
        if coords is not None:
            is_Unique = True
            for prev_artefact in self.artefacts_list:
                if prev_artefact == coords:
                    is_Unique = False
            if is_Unique:
                self.artefacts_list.append(coords)
                self.artefact_x_y = coords
                rospy.logerr(f"NEW: Found artifact at X: {self.artefact_x_y[0]:.2f}, Y: {self.artefact_x_y[1]:.2f}, Theta: {self.artefact_x_y[2]:.2f}")
                self.exploration_state_ = PlannerType.OBJECT_IDENTIFIED_SCAN
    ###########################################################################################################################
    
    def main_loop(self):
        while not rospy.is_shutdown():
            # Get the current status, possible statuses here: https://docs.ros.org/en/noetic/api/actionlib_msgs/html/msg/GoalStatus.html
            action_state = self.move_base_action_client_.get_state()
            print("Current State:", self.exploration_state_.name)
            self.exploration_planner(self.exploration_state_)

            # Delay so the loop doesn't run too fast
            rospy.sleep(0.1)


    def exploration_planner(self, action_state):

        if self.exploration_state_ == PlannerType.WAITING_FOR_MAP:
            print('entering .... WAITING for map')
            self.handle_waiting_for_map()

        elif self.exploration_state_ == PlannerType.SELECTING_FRONTIER or self.exploration_state_ == PlannerType.HANDLE_REJECTED_FRONTIER:
            print('entering ....SELECTING frontiers')
            self.handle_selecting_frontier()

        elif self.exploration_state_ == PlannerType.MOVING_TO_FRONTIER:
            print('entering ....MOVING TO frontiers')
            self.handle_moving_to_frontier(action_state)

        elif self.exploration_state_ == PlannerType.HANDLE_TIMEOUT:
            print('entering ....HANDLE TIMEOUT')
            self.handle_timeout()

        elif self.exploration_state_ == PlannerType.EXPLORED_MAP:
            rospy.loginfo("Exploration completed successfully.")
            rospy.sleep(1.0)

        elif self.exploration_state_ == PlannerType.OBJECT_IDENTIFIED_SCAN:
            rospy.loginfo("Object identified.")
            self.object_identified_scan()
               
     
    def handle_waiting_for_map(self):
        while self.current_map_ is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(0.5)  # Wait for the map to be received
        print ('map acquired')  
        self.exploration_state_ = PlannerType.SELECTING_FRONTIER
                
            
    def handle_selecting_frontier(self):
        print('frontiers found selecting frontiers')
        
        frontier_points = self.find_frontiers()
        self.group_frontiers(frontier_points)

        if self.selected_frontier is None:
            rospy.logwarn('No frontier selected.')
            self.exploration_state_ = PlannerType.EXPLORED_MAP
        else:
            rospy.loginfo('Frontier selected')
            

        self.exploration_state_ = PlannerType.MOVING_TO_FRONTIER
        
    # GROUPING POINTS ##############################################################################
    def find_frontiers(self):
        frontier_points = []
        # Iterate over the occupancy grid
        for y in range(self.MAP_HEIGHT):
            for x in range(self.MAP_WIDTH):
                value = self.occupancy_grid[y * self.MAP_WIDTH + x]

                # Known point (0-100) with unknown neighbors is a frontier
                if 0 <= value <= self.INTENSITY_THRESHOLD and self.has_unknown_neighbor(x, y):
                    frontier_points.append((x, y))

        return frontier_points
    
    # Check all neighbours of the cell to see if its unknown - not check diagonals atm - 
    # If at least one of the neighbours is unknown, return true
    def has_unknown_neighbor(self, x, y):
        neighbors = [   (x - 1, y),     # Left
                        (x + 1, y),     # Right
                        (x, y - 1),     # Down
                        (x, y + 1) ]     # Up
        
        for nx, ny in neighbors:
            if 0 <= nx < self.MAP_WIDTH and 0 <= ny < self.MAP_HEIGHT:
                index = ny * self.MAP_WIDTH + nx
                if self.occupancy_grid[index] == -1:  # Unknown
                    return True
                
        return False
    
        
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
        for points in self.frontier_groups:
            n = len(points)
            if n > self.MIN_CLUSTER_POINTS:  # Checks if the group has more than 50 points
                # Calculate the average x and y
                avg_x = sum(point[0] for point in points) / n
                avg_y = sum(point[1] for point in points) / n
                # Store the average point as a tuple in the average_points list
                self.average_and_size_frontier_points.append(((avg_x, avg_y), n))
    #####################################################################################################


    def handle_moving_to_frontier(self, action_state):
        """
        Handle robot navigation to frontiers while monitoring for object detection.
        Cancels navigation goals if an object is detected.
        """
        # Ensure there is a selected frontier to attempt
        while self.selected_frontier and (self.exploration_state_ != PlannerType.OBJECT_IDENTIFIED_SCAN):
            try:
                self.send_goal_simple(self.planner_move_to_frontier())
   
                # Take the current frontier to move towards
                frontier = self.selected_frontier[0]
                
                # Send the goal and start monitoring
                self.send_goal(frontier)
                rospy.loginfo('Moving to frontier')
                start_time = rospy.Time.now()
                timeout_duration = rospy.Duration(CaveExplorer.TIME_OUT_MAX)  # 15-second timeout
                
                # Monitor goal progress
                while not rospy.is_shutdown():
                    # Check if we've exceeded our timeout
                    if rospy.Time.now() - start_time >= timeout_duration:
                        rospy.loginfo("Goal timeout reached")
                        self.move_base_action_client_.cancel_goal()
                        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                        return
                    
                    # Check if object was detected
                    if self.exploration_state_ == PlannerType.OBJECT_IDENTIFIED_SCAN:
                        rospy.loginfo("Object detected - cancelling navigation goal")
                        self.move_base_action_client_.cancel_goal()
                        # Wait for cancellation to complete
                        rospy.sleep(0.5)
                        return
                    
                    # Get current goal state
                    action_state = self.move_base_action_client_.get_state()
                    
                    # Handle different goal states
                    if action_state == actionlib.GoalStatus.SUCCEEDED:
                        rospy.loginfo("Successfully reached the frontier!")
                        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                        return
                    
                    elif action_state in {actionlib.GoalStatus.REJECTED, 
                                        actionlib.GoalStatus.ABORTED,
                                        actionlib.GoalStatus.PREEMPTED}:
                        rospy.loginfo(f"Goal failed with state: {action_state}")
                        self.exploration_state_ = PlannerType.HANDLE_REJECTED_FRONTIER
                        return
                    
                    # Brief sleep to prevent CPU hogging
                    rospy.sleep(0.1)
                    
            except Exception as e:
                rospy.logerr(f"Error during frontier navigation: {str(e)}")
                self.move_base_action_client_.cancel_goal()
                self.exploration_state_ = PlannerType.WAITING_FOR_MAP
                return
            
        # If we exit the while loop, we've run out of frontiers
        if not self.selected_frontier:
            rospy.loginfo("No more frontiers to explore")
            self.exploration_state_ = PlannerType.EXPLORED_MAP


    # Evaluates the best frontier to move towards, and sends that goal to the action server to move the robot to that location
    # Frontiers are evalulated based on their average position, and their distance from the current position.
    def planner_move_to_frontier(self, action_state):
        # Find the closest frontier cluster
        if action_state != actionlib.GoalStatus.ACTIVE and self.average_and_size_frontier_points:

            robot_pose = self.get_pose_2d()
            target = []
            min_cost = float('inf')
            for group in self.average_and_size_frontier_points:
                cost= self.group_cost(robot_pose, group)
                #print("Cost: ", cost)
                if cost < min_cost:
                    min_cost = cost
                    (x, y), n = group
                    target = (x, y)

            x = target[0] * 0.1 - 10
            y = target[1] * 0.1 - 10
            theta = random.uniform(0, 2*math.pi)

            return (x, y, theta)
            
            
    # Returns the cost of a group in terms of it's cluster size and distance. Lower costs are preferable 
    # PASS THE FRICKEN AVERAGES
    # MAYBE: # Cost = Distance^2 + weight*(cluster size)
    def group_cost(self, current_position, group):
        (avg_x, avg_y), n = group
        distance = math.hypot((current_position.x - avg_x), (current_position.y - avg_y))
        # Distance should be > 1, Length should be < 1
        cost = self.DIST_WEIGHT * distance + self.LENGTH_WEIGHT * n

        return cost

            
    def handle_timeout(self):
        rospy.loginfo("@ handle_timeout")
        self.exploration_state_ = PlannerType.WAITING_FOR_MAP
   
        
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
        
        
    def send_goal_simple(self, coordinate):
        pose_2d = Pose2D
        pose_2d.x = coordinate[0]
        pose_2d.y = coordinate[1]
        pose_2d.theta = coordinate[2]
        # print(f'x:{pose_2d.x} , y:{pose_2d.y}')

        # Send a goal to "move_base" with "self.move_base_action_client_"
        action_goal = MoveBaseActionGoal()

        action_goal.goal.target_pose.header.frame_id = "map"
        action_goal.goal_id = self.goal_counter_
        self.goal_counter_ = self.goal_counter_ + 1
        action_goal.goal.target_pose.pose = pose2d_to_pose(pose_2d)

        # sending the goal to move base
        self.move_base_action_client_.send_goal(action_goal.goal)

                  
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
            
            if action_state in {actionlib.GoalStatus.REJECTED,
                                actionlib.GoalStatus.ABORTED,
                                actionlib.GoalStatus.PREEMPTED}:
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

        
if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('cave_explorer', anonymous=True)
    # Create the cave explorer
    cave_explorer = CaveExplorer()
    # Loop forever while processing callbacks
    cave_explorer.main_loop()


