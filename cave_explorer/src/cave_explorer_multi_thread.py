
class CaveExplorer:
    

        # Depth and scan data handling
        self.depth_data_ = None
        self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ SE3(0.005, 0.028, 0.013)
        self.scan_lock = threading.Lock()
        self.scan_data = None

        self.pool = mp.Pool(processes=mp.cpu_count())
        self._chunk_size = 1000 


    @staticmethod
    def _process_chunk(chunk_data, width):
        """Process a single chunk of the map to find frontiers"""
        chunk_array, start_row = chunk_data
        frontiers = set()
        
        # Create binary masks for free and unknown space
        free_space = chunk_array == 0
        unknown_space = chunk_array == -1

        # Define neighbor offsets
        shifts = [(0,1), (0,-1), (1,0), (-1,0)]

        # Process each shift direction
        for dx, dy in shifts:
            # Create shifted arrays
            shifted_free = np.roll(np.roll(free_space, dx, axis=0), dy, axis=1)
            shifted_unknown = np.roll(np.roll(unknown_space, dx, axis=0), dy, axis=1)
            
            # Find frontier cells
            frontier_cells = np.where(free_space & shifted_unknown)
            
            # Adjust row indices based on chunk position
            adjusted_frontiers = zip(
                frontier_cells[0] + start_row, 
                frontier_cells[1]
            )
            
            # Add to frontier set
            frontiers.update(adjusted_frontiers)

        return frontiers


    def image_callback(self, image_msg):
        classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]

        # Convert the ROS image message to a CV2 image
        cv_image = self.cv_bridge_.imgmsg_to_cv2(image_msg, "bgr8")

        # Process the image using YOLO
        # print('------------------------------------------------------------')
        # print('USING CUDA:', self.device_)
        # print('------------------------------------------------------------')
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
                    # rospy.loginfo(f"Center of {classes[class_id]}: ({center_x}, {center_y})")

                    # Get the 3D coordinates
                    art_xyz = self.get_posed_3d(center_x, center_y)

                    # Check if art_xyz is None before accessing its elements
                    if art_xyz is not None:
                        # Draw rectangle and label
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

                        # Detecting Mineral and Mushroom
                        if class_id == 1 or class_id == 4:
                            if class_id == 1:
                                artefact_list = self.mineral_artefacts
                            else:
                                artefact_list = self.mushroom_artefacts
                            
                            # Check if it doesn't already exist
                            already_exists = False
                            _art_art_dist_threshold = 7  # Minimum distance threshold between artefacts
                            for artefact in artefact_list:
                                if math.hypot(artefact[0] - art_xyz[0], artefact[1] - art_xyz[1]) > _art_art_dist_threshold:
                                    continue
                                else:
                                    already_exists = True
                                    break

                            if not already_exists:
                                artefact_list.append(art_xyz)

                                # # Start the go_to_artifact in a new thread to allow image_callback to continue
                                # nav_thread = threading.Thread(target=self.go_to_artifact, args=(art_xyz,))
                                # nav_thread.start()
                    else:
                        # Draw rectangle and label with warning color (red)
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        # rospy.logwarn("Could not retrieve 3D coordinates.")

        # Convert the modified CV2 image back to a ROS Image message
        processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")

        # Publish the processed image
        self.image_pub_.publish(processed_msg)    

    def map_callback(self, map_msg):
        if self.current_map_ is None:
            rospy.loginfo("Map received")
            self.current_map_ = map_msg  
        elif self.current_map_.header.stamp != map_msg.header.stamp:
            rospy.loginfo("Map updated")
            self.current_map_ = map_msg
        else:
            rospy.loginfo("Map not updated")
                                
    
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
            
        # rospy.sleep(0.5)
                
                
    def object_identified_scan(self):
        # Stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub_.publish(twist)
        
        # identify object coordinates and move towards it
        # align so that the object is in the center of the camera
        # move towards the object and 
        
        # Return to the exploration state
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        
        # rospy.sleep(0.5)


    def handle_waiting_for_map(self):
        print ( 'waiting for map.....')
        while self.current_map_ is None:
            rospy.logwarn("Map not available yet, waiting for map...")
            rospy.sleep(1.0)  # Wait for the map to be received
            continue  # Keep waiting until the map is received
        # load the current map
        current_map = self.current_map_
        print ( ' map acquired')  
        self.exploration_state_ = ExplorationsState.IDENTIFYING_FRONTIERS
        print ( 'state changed to identifying frontiers')
        
        # rospy.sleep(2)


    def handle_identifying_frontiers(self):
        print ( 'identifying frontiers,,,,,')
        frontiers = self.identify_frontiers(self.current_map_)
        if not frontiers:
            rospy.loginfo('No frontiers found!')
            self.exploration_state_ = ExplorationsState.EXPLORED_MAP
        else:
            self.exploration_state_ = ExplorationsState.SELECTING_FRONTIER
            
        # rospy.sleep(2)


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
            
        # rospy.sleep(2)


    def handle_moving_to_frontier(self, action_state):
        
        ## as the robot moves to the frontier, it will check object detection
        ## if object detected, it will stop and move towards the object
        
        ## implement logic for changing planner to object detection
        
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
            
            
        # rospy.sleep(2)
            
            
    def handle_timeout(self):
        rospy.loginfo("Timeout reached.")
        self.current_map_ = None
        self.selected_frontier = None
        self.exploration_state_ = ExplorationsState.WAITING_FOR_MAP
        
        # rospy.sleep(0.5)

                                        
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
        # rospy.sleep(0.1)
        
        return frontiers


    def get_neighbors(self, row, col, map_array):
        neighbors = []
        height, width = map_array.shape
        if row > 0: neighbors.append((row-1, col))
        if row < height-1: neighbors.append((row+1, col))
        if col > 0: neighbors.append((row, col-1))
        if col < width-1: neighbors.append((row, col+1))
        rospy.sleep(0.0000000001)
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
        return np.hypot((point1[0] - point2[0]), (point1[1] - point2[1]))
        
        
    def compute_distance(self, frontier):
        map_res = self.current_map_.info.resolution
        map_origin = self.current_map_.info.origin.position
        x, y = frontier
        frontier_x = x * map_res + map_origin.x
        frontier_y = y * map_res + map_origin.y
        distance = np.hypot((frontier_x - self.current_position.x ), (frontier_y - self.current_position.y ))
        return distance
        
        
    def is_valid_frontier(self, frontier_point):
        # """ Check if the frontier point is far enough from all visited frontiers. """
        min_distance = 0 # Define a minimum distance threshold
        
        for visited_point in self.visited_frontiers:
            # Calculate Euclidean distance between the frontier and visited point
            distance = np.hypot((frontier_point[0] - visited_point[0]), (frontier_point[1] - visited_point[1]))
            if distance < min_distance:
                return False  # The frontier is too close to a visited point
            
        # rospy.sleep(0.5)
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
        
        # rospp.sleep(0.5)
            
        
    def index_to_position(self, index):
        # Convert the index of the grid cell to a (x, y) position in the map
        width = self.current_map_.info.width
        resolution = self.current_map_.info.resolution
        origin_x = self.current_map_.info.origin.position.x
        origin_y = self.current_map_.info.origin.position.y
        
        x = (index % width) * resolution + origin_x
        y = (index // width) * resolution + origin_y
        
        # rospy.sleep(0.5)
        
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
            elif (self.planner_type_ == PlannerType.EXPLORATION) and (action_state == actionlib.GoalStatus.ACTIVE):
                print("Exploration preempted!")
                self.exploration_done_ = False
                action_state = actionlib.GoalStatus.PREEMPTING

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
            rospy.sleep(0.5)


if __name__ == '__main__':

    # Create the ROS node
    rospy.init_node('cave_explorer')

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    # Loop forever while processing callbacks
    cave_explorer.main_loop()