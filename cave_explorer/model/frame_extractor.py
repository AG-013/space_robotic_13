# This script extracts one frame per second out of a rosbag recording
# Saved frames can be used for training a model once annotated

import rosbag
import cv2
from cv_bridge import CvBridge
import os
import rospy

def extract_images_from_bag(bag_file, topic_name, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bridge = CvBridge()
    last_saved_time = None
    image_count = 0

    with rosbag.Bag(bag_file, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[topic_name]):
            current_time = msg.header.stamp.to_sec()

            # Save the image if it's at least 1 second since the last saved image
            if last_saved_time is None or current_time - last_saved_time >= 1.0:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                image_filename = os.path.join(output_dir, f'image_{image_count:04d}.png')
                cv2.imwrite(image_filename, cv_image)
                print(f'Saved {image_filename}')
                last_saved_time = current_time
                image_count += 1

if __name__ == '__main__':
    bag_file = 'bag.bag'
    topic_name = '/camera/rgb/image_raw'
    output_dir = 'output'

    extract_images_from_bag(bag_file, topic_name, output_dir)
