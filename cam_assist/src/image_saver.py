#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import os
import sys
from datetime import datetime


class ImageSaver:
    def __init__(self, folder_name):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        # Define the base save directory
        base_dir = "/home/selimon/sr_ws/src/cam_assist/src/captured_images"
        
        # Combine the base directory with the folder name passed as an argument
        self.save_dir = os.path.join(base_dir, folder_name)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo(f"Created directory: {self.save_dir}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Generate a unique filename using timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = os.path.join(self.save_dir, f"image_{timestamp}.jpg")
        
        # Save the image
        success = cv2.imwrite(image_filename, cv_image)
        if success:
            rospy.loginfo(f"Saved image as {image_filename}")
            # Shutdown the node after saving a single image
            rospy.signal_shutdown("Image saved")
        else:
            rospy.logerr(f"Failed to save image: {image_filename}")


if __name__ == '__main__':
    rospy.init_node('image_saver', anonymous=True)

    # Check if a folder name is provided as an argument
    if len(sys.argv) < 2:
        rospy.logerr("Please provide a folder name as an argument.")
        sys.exit(1)

    # Get the folder name from the command-line argument
    folder_name = sys.argv[1]

    image_saver = ImageSaver(folder_name)
    rospy.spin()