
    # def create_marker(self, art_xyz, marker_id, r, g, b):
    #     marker = Marker()
    #     marker.header.frame_id = "map"
    #     marker.header.stamp = rospy.Time.now()
    #     marker.ns = "artifact_marker"
    #     marker.id = marker_id
        
    #     marker.type = Marker.SPHERE
    #     marker.action = Marker.ADD
    #     marker.pose.position.x = art_xyz[0]
    #     marker.pose.position.y = art_xyz[1]
    #     marker.pose.position.z = art_xyz[2]
    #     marker.scale.x = 1
    #     marker.scale.y = 1
    #     marker.scale.z = 1
    #     marker.color.a = 1.0
    #     marker.color.r = r
    #     marker.color.g = g
    #     marker.color.b = b
    #     marker.lifetime = rospy.Duration(0.6)
        
    #     # Add orientation (required for proper visualization)
    #     marker.pose.orientation.w = 1.0
        
    #     return marker
    # def publish_artefact_markers(self, event):
    #     marker_id = 0  # Start counter for unique IDs
        
    #     # Publish mineral markers (green)
    #     for art_xyz in self.mineral_artefacts:
    #         marker = self.create_marker(art_xyz, marker_id, 0.004, 0.195, 0.125)
    #         self.marker_pub.publish(marker)
    #         marker_id += 1  # Increment for the next marker

    #     # Publish mushroom markers (grey)
    #     for art_xyz in self.mushroom_artefacts:
    #         marker = self.create_marker(art_xyz, marker_id, 0.5, 0.5, 0.5)
    #         self.marker_pub.publish(marker)
    #         marker_id += 1  # Increment for the next marker


    # def go_to_artifact(self, artefact_trans: tuple):
    #     # Navigate robot to target artifact position

    #     # Constants
    #     ANGLE_THRESHOLD = 0.1  # Radians
    #     DISTANCE_THRESHOLD = 1.75  # m
    #     LINEAR_SPEED = 0.5  # m/s
    #     ANGULAR_SPEED = 0.5  # rad/s
    #     REVERSE_DISTANCE = 1.0  # m to reverse after reaching the artifact

    #     # 1. Initial Rotation to face target
    #     pose_2d = self.get_pose_2d()
    #     target_theta = math.atan2(artefact_trans[1] - pose_2d.y, artefact_trans[0] - pose_2d.x)
    #     cur_theta = wrap_angle(pose_2d.theta)

    #     # Store the original angle before rotation to face the artifact
    #     original_theta = cur_theta

    #     while abs(wrap_angle(target_theta - cur_theta)) > ANGLE_THRESHOLD:
    #         rospy.logerr('Rotating Robot to face artifact')
    #         pose_2d = self.get_pose_2d()
    #         cur_theta = wrap_angle(pose_2d.theta)
            
    #         twist_msg = Twist()
    #         angle_diff = wrap_angle(target_theta - cur_theta)
    #         twist_msg.angular.z = ANGULAR_SPEED * angle_diff
            
    #         self.cmd_vel_pub_.publish(twist_msg)
            
    #     # Stop rotation
    #     self.cmd_vel_pub_.publish(Twist())
    #     rospy.sleep(0.5)  # Brief pause after rotation

    #     # 2. Forward Movement
    #     pose_2d = self.get_pose_2d()
    #     distance = self.get_distance_to_target(artefact_trans, pose_2d)

    #     while distance > DISTANCE_THRESHOLD and not rospy.is_shutdown():
    #         while self.about_to_collide():
    #             # Avoid obstacles while moving toward the artifact
    #             twist_msg = Twist()
    #             if self.move_left():
    #                 rospy.logerr(f'ABOUT TO COLLIDE SO MOVING LEFT: {distance:.2f}m')
    #                 twist_msg.linear.y = 0.5
    #             else:
    #                 rospy.logerr(f'ABOUT TO COLLIDE SO MOVING RIGHT: {distance:.2f}m')
    #                 twist_msg.linear.y = -0.5
    #             self.cmd_vel_pub_.publish(twist_msg)
            
    #         rospy.logerr(f'Moving forward, distance: {distance:.2f}m')
    #         pose_2d = self.get_pose_2d()
            
    #         # Recalculate target angle and current angle difference
    #         target_theta = math.atan2(artefact_trans[1] - pose_2d.y, artefact_trans[0] - pose_2d.x)
    #         angle_diff = wrap_angle(target_theta - wrap_angle(pose_2d.theta))
            
    #         twist_msg = Twist()
    #         twist_msg.linear.x = LINEAR_SPEED
    #         twist_msg.angular.z = 0.5 * ANGULAR_SPEED * angle_diff
            
    #         self.cmd_vel_pub_.publish(twist_msg)
            
    #         # Update distance
    #         distance = self.get_distance_to_target(artefact_trans, pose_2d)
    #         rospy.sleep(0.1)

    #     # Stop the robot after reaching the artifact
    #     self.cmd_vel_pub_.publish(Twist())
    #     rospy.sleep(2)
    #     # rospy.loginfo('Arrived at the artifact')
    #     rospy.loginfo("\033[93mArrived at the artifact\033[0m")

    #     # 3. Reverse movement
    #     rospy.loginfo('Reversing away from the artifact')
    #     reverse_distance = 0
    #     while reverse_distance < REVERSE_DISTANCE and not rospy.is_shutdown():
    #         twist_msg = Twist()
    #         twist_msg.linear.x = -LINEAR_SPEED  # Move backward
    #         self.cmd_vel_pub_.publish(twist_msg)
            
    #         rospy.sleep(0.1)
    #         reverse_distance += LINEAR_SPEED * 0.1  # Approximate how much the robot has moved

    #     # Stop the robot after reversing
    #     self.cmd_vel_pub_.publish(Twist())
    #     rospy.loginfo('Reversed away from the artifact')

    #     # 4. Rotate back to original orientation
    #     rospy.loginfo('Rotating back to the original orientation')
    #     cur_theta = wrap_angle(self.get_pose_2d().theta)
    #     while abs(wrap_angle(original_theta - cur_theta)) > ANGLE_THRESHOLD and not rospy.is_shutdown():
    #         cur_theta = wrap_angle(self.get_pose_2d().theta)
            
    #         twist_msg = Twist()
    #         angle_diff = -wrap_angle(original_theta - cur_theta)
    #         twist_msg.angular.z = ANGULAR_SPEED * angle_diff
    #         self.cmd_vel_pub_.publish(twist_msg)
    #         rospy.sleep(0.1)

    #     # Stop rotation
    #     self.cmd_vel_pub_.publish(Twist())
    #     rospy.loginfo('Returned to original orientation')

 
    # def about_to_collide(self):
    #     # Check if the robot is about to collide based on the latest laser scan data.
    #     SAFETY_DISTANCE = 0.85  # Set safety distance threshold (in meters)
        
    #     # Ensure thread-safe access to laser scan data
    #     with self.scan_lock:
    #         if self.scan_data is None:
    #             rospy.logwarn("Laser scan data is not available")
    #             return False  # No data, assume no collision risk

    #         # Determine the forward sector to check (e.g., front 60 degrees)
    #         # Get the indices for the front sector (centered in the middle of the scan data)
    #         # Extract the relevant ranges for the front sector
    #         front_ranges_left = self.scan_data[0:60]
    #         front_ranges_right = self.scan_data[300:]
    #         front_ranges = front_ranges_left + front_ranges_right
    #         valid_ranges = [r for r in front_ranges if not math.isinf(r) and not math.isnan(r)]

    #         if not valid_ranges:
    #             rospy.logwarn("No valid laser scan data in the forward sector")
    #             return False  # No valid data, assume no collision risk

    #         # Check if any obstacle is within the safety distance
    #         if min(valid_ranges) < SAFETY_DISTANCE:
    #             rospy.logerr("Obstacle detected within safety distance!")
    #             return True  # Collision likely
    #         else:
    #             rospy.logerr(f"NO Obstacle detected SAFE TO MOVE! {min(valid_ranges)}")
    #             return False  # No collision risk
    
    
    # Checks if robot can turn left or right                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    # def move_left(self):
    #     # Check if the robot is about to collide based on the latest laser scan data.
    #     SAFETY_DISTANCE = 1.0  # Set safety distance threshold (in meters)
        
    #     # Ensure thread-safe access to laser scan data
    #     with self.scan_lock:
    #         if self.scan_data is None:
    #             rospy.logwarn("Laser scan data is not available")
    #             return False  # No data, assume no collision risk

    #         # Determine the forward sector to check (e.g., front 60 degrees)
    #         # Get the indices for the front sector (centered in the middle of the scan data)
    #         # Extract the relevant ranges for the front sector
    #         ranges_left = self.scan_data[60:120]
    #         valid_ranges = [r for r in ranges_left if not math.isinf(r) and not math.isnan(r)]

    #         if not valid_ranges:
    #             rospy.logwarn("move_left(): No valid laser scan data in the left sector, so just gonna turn right")
    #             return False  # No valid data, assume no collision risk

    #         # Check if any obstacle is within the safety distance
    #         if min(valid_ranges) < SAFETY_DISTANCE:
    #             rospy.logerr("move_left(): Obstacle detected within safety distance!, so just gonna turn right")
    #             return False  # Collision likely
    #         else:
    #             rospy.logerr(f"move_left(): NO Obstacle detected SAFE TO MOVE LEFT! {min(valid_ranges)}")
    #             return True  # No collision risk

# 5555555555555555555555555555555555555555555555555555555555555555555555555555555555555

        
    # def odom_callback(self,odom_sub_):
    #     # Extract position data
    #     # rospy.loginfo ("Odo received")
    #     self.odom_ = odom_sub_

    #     self.current_position = Point()

    #     # Extract pose
    #     position = self.odom_.pose.pose.position
    #     self.current_position.x = position.x
    #     self.current_position.y = position.y

    #     orientation = self.odom_.pose.pose.orientation
    #     # Extract twist 
    #     linear = self.odom_.twist.twist.linear
    #     angular = self.odom_.twist.twist.angular
        
    #     # rospy.sleep(0.5)
        
  
    
    # def select_nearest_frontier(self, frontiers):
    #     frontiers_merged = frontiers
            
    #     # rospy.loginfo(f'Frontiers found: {len(frontiers)}')
    #     frontiers_merged= self.merge_close_frontiers_to_one(frontiers)
        
    #     ### use a weighted frontier selection process 
    #      ### based on distance alpha 1 , - size of frontier alpha 2, + orientation towards the frontier alpha 3
    #      ## alpha 1 = 0.5, alpha 2 = 0.3, alpha 3 = 0.2
        
        
    #     # rospy.loginfo(f'Frontiers merged: {len(frontiers_merged)}')
    #     # frontiers_merged.sort(key=self.compute_distance)
    #     # print ( f'frontiers sorted distance: {len(frontiers_merged)}')
    #     # rospy.loginfo(f'Frontiers sorted: {len(frontiers_merged)}')
    #     # frontiers_merged =self.is_valid_frontier(frontiers_merged)
    #     # print ( f'frontiers sorted distance + is valid: {len(frontiers_merged)}')
    #     # print ( 'frontiers sorted')
    #     # rospy.sleep(2)
    #     return frontiers_mergedx
