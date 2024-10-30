#!/usr/bin/env python3

import os
import threading
import time

# Math Modules
from spatialmath import SE3
# May have to install these packages:
# pip install roboticstoolbox-python
# pip install spatialmath-python
import math
import numpy as np

# Machine Learning / OpenCV Modules
import cv2 # OpenCV2
import torch
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError

# ROS Modules
import tf
import rospy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker

from helper_functions import *


class ArtefactLocator:
    # Class constants - define these at class level
    CONFIDENCE_THRESHOLD = 0.65
    ARTIFACT_DISTANCE_THRESHOLD = 7.0
    TRANSFORM_TIMEOUT = 10.0  # seconds
    
    def __init__(self):
        # Initialize tf listener first
        self.tf_listener_ = tf.TransformListener()
        
        # Initialize CvBridge
        self.cv_bridge_ = CvBridge()
        
        # Now wait for the transform
        rospy.loginfo("Waiting for transform from map to base_link")
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.tf_listener_.canTransform("map", "base_link", rospy.Time(0.)):
                break
            if (rospy.Time.now() - start_time).to_sec() > self.TRANSFORM_TIMEOUT:
                raise RuntimeError("Transform timeout - SLAM node not available")
            rospy.sleep(0.1)
            rospy.logwarn_throttle(1, "Waiting for transform... Have you launched a SLAM node?")

        # Initialize YOLO model
        self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        path = os.path.abspath(__file__)
        src_dir = os.path.dirname(path)
        parent_dir = os.path.abspath(os.path.join(src_dir, '..', '..'))
        model_path = os.path.join(parent_dir, 'cam_assist/src/test_train/yolov11s_trained_optimized.pt')
        self.model_ = YOLO(model_path)
        rospy.loginfo(f"Using device: {self.device_}")

        # Subscribe to the camera topic
        self.image_sub_ = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback, queue_size=1)
        self.depth_sub_ = rospy.Subscriber("/camera/depth/points", PointCloud2, self.depth_callback, queue_size=1)

        # Publisher for the camera detections
        self.image_pub_ = rospy.Publisher("/detections_image", Image, queue_size=5)
        self.marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
                
        # For depth
        self.depth_data_ = None
                
        # For Transformation
        self.base_2_depth_cam = SE3(0.5, 0, 0.9) @ SE3(0.005, 0.028, 0.013)
                        
        self.marker_timer = rospy.Timer(rospy.Duration(0.5), self.publish_artefact_markers)
        
        self.mineral_artefacts = []
        self.mushroom_artefacts = []
        
        
    def image_callback(self, image_msg):
        classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]

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
                if confidence >= ArtefactLocator.CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    class_id = int(box.cls.item())  # Get the class ID from the tensor
                    label = f'{classes[class_id]} {confidence:.2f}'  # Class name and confidence

                    # Calculate and print center point of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    rospy.loginfo(f"Center of {classes[class_id]}: ({center_x}, {center_y})")

                    # Get the 3D coordinates
                    art_xyz = self.get_posed_3d(center_x, center_y)
                    
                    # Check if art_xyz is None before accessing its elements
                    if art_xyz is not None:
                        # rospy.loginfo(f"X: {art_xyz[0]},  Y: {art_xyz[1]},  Z: {art_xyz[2]}")
                        print(art_xyz)
                        
                        # Draw rectangle and label
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(cv_image, (center_x, center_y), 5, (0, 0, 255), -1)

                        # classes = ["Alien", "Mineral", "Orb", "Ice", "Mushroom", "Stop Sign"]
                        # Detecting Mineral and Mushroom - choosing mushroom because mineral and mushroom aren't together, so less points of error                         
                        if class_id == 1 or class_id == 4:
                            if class_id == 1:
                                artefact_list = self.mineral_artefacts
                            else:
                                artefact_list = self.mushroom_artefacts
                            # Check if it doesn't already exist first
                            already_exists = False
                            for artefact in artefact_list:
                                        
                                if math.hypot(artefact[0] - art_xyz[0], artefact[1] - art_xyz[1]) > ArtefactLocator.ARTIFACT_DISTANCE_THRESHOLD:
                                    continue
                                else:
                                    already_exists = True
                                    break
                            if not already_exists:
                                artefact_list.append(art_xyz)
                    else:
                        # Draw rectangle and label
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        rospy.logwarn("Could not retrieve 3D coordinates.")

        # Convert the modified CV2 image back to a ROS Image message
        try:
            processed_msg = self.cv_bridge_.cv2_to_imgmsg(cv_image, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Publish the processed image
        self.image_pub_.publish(processed_msg)
        # rospy.loginfo("Published processed image")
        
        
    def depth_callback(self, depth_msg):
        try:
            self.depth_data_ = depth_msg
        except Exception as e:
            rospy.logwarn(f"Error in depth callback: {e}")
            
            
    # Cons of this code: only does one pixel - Since depth data is noisy it is essential to get nearby pixels, and average them    
    # Given pix_xy (a tuple), extract the 3d position of the pixels by applying a transformation matrix of the current pose * matrix of relative pose    
    # Extract the 3D coordinates from PointCloud2 data at pixel (pixel_x, pixel_y)        
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
        theta = wrap_angle(2.0 * math.acos(qw) if qz >= 0 else -2.0 * math.acos(qw))
        
        # Create robot pose transformation
        robot_pos = SE3(x, y, z) @ SE3.Rz(theta)  # Using Rz for 2D rotation instead of RPY

        # Extract point from depth data
        point = list(point_cloud2.read_points(  self.depth_data_, 
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
        
        
    def publish_marker(self, art_xyz, marker_id, marker_type, color_r, color_g, color_b):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "artifact_marker"
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = art_xyz[0]
        marker.pose.position.y = art_xyz[1]
        marker.pose.position.z = art_xyz[2]
        marker.scale.x = 1
        marker.scale.y = 1
        marker.scale.z = 1
        marker.color.a = 1.0
        marker.color.r = color_r
        marker.color.g = color_g
        marker.color.b = color_b
        marker.lifetime = rospy.Duration(0.6)
        marker.pose.orientation.w = 1.0
        self.marker_pub.publish(marker)
    def publish_artefact_markers(self, event):
        marker_id = 0  # Start counter for unique IDs

        # Publish markers for mineral artefacts
        for art_xyz in self.mineral_artefacts:
            self.publish_marker(art_xyz, marker_id, Marker.SPHERE, 0.004, 0.195, 0.125)
            marker_id += 1

        # Publish markers for mushroom artefacts
        for art_xyz in self.mushroom_artefacts:
            self.publish_marker(art_xyz, marker_id, Marker.CYLINDER, 1.0, 0.0, 0.0)
            marker_id += 1


if __name__ == '__main__':
    rospy.init_node('artefact_locator_node')
    artefact_locator = ArtefactLocator()
    rospy.spin()