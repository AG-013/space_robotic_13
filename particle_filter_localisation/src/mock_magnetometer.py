#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np

def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def random_normal(stddev):
    # Returns a random number with normal distribution, 0 mean and a standard deviation of "stddev"
    return np.random.normal(0.0, stddev)


class Magnetometer:
    def __init__(self):

        # Parameters
        self.noise_stddev_ = 0.349066 # 20 degrees
        self.msg_time_ = rospy.get_rostime()
        self.msg_period_ = rospy.Duration(1.)

        # Subscribers
        self.pose_sub_ = rospy.Subscriber('/base_pose_ground_truth', Odometry, self.pose_callback, queue_size=1) # Subscribes to ground truth location

        # Publishers
        self.compass_pub_ = rospy.Publisher('compass', Float32, queue_size=1)



    def pose_callback(self, odom_msg):
        # Receive an odometry message

        if ((rospy.get_rostime() - self.msg_time_) >= self.msg_period_):

            self.msg_time_ = rospy.get_rostime()

            # Convert it to a heading in radians
            pose_theta = 2. * math.acos(odom_msg.pose.pose.orientation.w)

            if odom_msg.pose.pose.orientation.z < 0.:
                pose_theta = -pose_theta

            # Add Guassian noise
            pose_theta += random_normal(self.noise_stddev_)

            # Wrap it
            pose_theta = wrap_angle(pose_theta)

            # Publish the compass reading
            msg = Float32(pose_theta)
            self.compass_pub_.publish(msg)



if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('magnetometer')

    # Create the particle filter
    Magnetometer()

    # Loop forever while processing callbacks
    rospy.spin()


