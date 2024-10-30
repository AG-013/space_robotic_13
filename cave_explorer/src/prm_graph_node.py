#!/usr/bin/env python3

import rospy
import math
import tf
from geometry_msgs.msg import Pose2D, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker

from helper_functions import wrap_angle


class Node:
    def __init__(self, x, y, idx, link):
        # Index of the node in the graph
        self.idx = idx
        # Position of node
        self.x = x
        self.y = y
        self.link = link


class PRPM:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('prm_node', anonymous=True)

        # Wait for the transform to become available
        rospy.loginfo("Waiting for transform from map to base_link")
        self.tf_listener_ = tf.TransformListener()

        while not rospy.is_shutdown() and not self.tf_listener_.canTransform("map", "base_link", rospy.Time(0)):
            rospy.sleep(0.1)
            rospy.loginfo("Waiting for transform... Have you launched a SLAM node?")        
        rospy.loginfo("Accepted, node is running")        

        self.nodes_ = []
        self.potential_node = 0

        # Subscribers
        self.lidar_sub_ = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        
        # Publishers
        self.prm_graph_pub_ = rospy.Publisher('prm_graph', Marker, queue_size=10)


    def get_pose_2d(self):
        # Lookup the latest transform
        try:
            (trans, rot) = self.tf_listener_.lookupTransform('map', 'base_link', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn(f"Transform error: {e}")
            return None

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = trans[0]
        pose.y = trans[1]

        qw = rot[3]
        qz = rot[2]

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else: 
            pose.theta = wrap_angle(-2. * math.acos(qw))

        rospy.logdebug(f"Current pose: {pose}")
        return pose


    def scan_callback(self, scan_msg):
        obstructed = 0
        distance_threshold = 6  # How much free space is required to add a node
        distance_set = 4  # How far away from current position node should be
        indices_of_interest = [355, 356, 357, 358, 0, 1, 2, 3, 4]  # 10 degree window
        readings_in_range = [scan_msg.ranges[i] for i in indices_of_interest]

        # Check if target range is clear and suitable for a point in the graph
        for reading in readings_in_range:
            if reading < distance_threshold:
                obstructed = 1   
                break

        if obstructed == 1:
            self.potential_node = 0
        else:
            pose = self.get_pose_2d()
            if pose is not None:
                front_x = pose.x + distance_set * math.cos(pose.theta)
                front_y = pose.y + distance_set * math.sin(pose.theta)
                self.potential_node = Node(front_x, front_y, 0, 0)


    def update_prm(self):
        distance_threshold = 4  # How far away nodes should be
        distance = distance_threshold + 1
        node_num = len(self.nodes_)
        closest_node = 0
        
        pose = self.get_pose_2d()
        if pose is None:
            return

        # If no nodes exist, publish current position
        if node_num == 0:
            self.nodes_.append(Node(pose.x, pose.y, node_num, 0))
            self.publish_prm()
        else:
            # Find closest node to current position
            for node in self.nodes_:
                current_distance = math.hypot((pose.x - node.x), (pose.y - node.y))
                if current_distance < distance:
                    distance = current_distance
                    closest_node = node.idx

            # If the node is further than threshold, publish it
            if distance > distance_threshold:
                self.nodes_.append(Node(pose.x, pose.y, node_num, closest_node))
                self.publish_prm()
            # Alternatively, there may be a potential node in front of the robot
            elif self.potential_node != 0:
                distance = float('inf')
                proximity = 0
                # Check whether this node is suitable based on proximity to existing nodes
                for node in self.nodes_:
                    current_distance = math.sqrt((self.potential_node.x - node.x) ** 2 + (self.potential_node.y - node.y) ** 2)
                    if current_distance < distance:
                        distance = current_distance
                        closest_node = node.idx
                    if distance_threshold > current_distance:
                        proximity = 1
                if proximity == 0:
                    self.nodes_.append(Node(self.potential_node.x, self.potential_node.y, node_num, closest_node))
                    self.publish_prm()


    def publish_prm(self):
        node_num = len(self.nodes_) - 1

        # Graph Point publisher
        point_marker = Marker()
        point_marker.header.frame_id = "map"
        point_marker.header.stamp = rospy.Time.now()
        point_marker.ns = "points"
        point_marker.id = node_num
        point_marker.type = Marker.SPHERE
        point_marker.action = Marker.ADD

        point_marker.pose.position = Point(self.nodes_[node_num].x, self.nodes_[node_num].y, 0)
        point_marker.pose.orientation.w = 1.0
        point_marker.scale.x = point_marker.scale.y = point_marker.scale.z = 1
        point_marker.color.r = 1.0
        point_marker.color.b = 1.0
        point_marker.color.a = 1.0
        point_marker.lifetime = rospy.Duration(0)
        
        self.prm_graph_pub_.publish(point_marker)

        # Graph Line publisher
        if node_num > 0:
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "edges"
            line_marker.id = node_num
            line_marker.type = Marker.LINE_LIST
            line_marker.action = Marker.ADD

            # Connect nodes that are within 6 units
            for node in self.nodes_:
                current_distance = math.sqrt((self.nodes_[node_num].x - node.x) ** 2 + 
                                          (self.nodes_[node_num].y - node.y) ** 2)
                if current_distance < 6:
                    line_marker.points.append(Point(self.nodes_[node_num].x, self.nodes_[node_num].y, 0))
                    line_marker.points.append(Point(node.x, node.y, 0))

            line_marker.pose.orientation.w = 1.0
            line_marker.scale.x = line_marker.scale.y = line_marker.scale.z = 0.25
            line_marker.color.r = 0.5
            line_marker.color.b = 0.5
            line_marker.color.a = 1.0
            line_marker.lifetime = rospy.Duration(0)
            
            self.prm_graph_pub_.publish(line_marker)


    def run(self):
        """Main run loop"""
        rate = rospy.Rate(5)  # 5Hz
        while not rospy.is_shutdown():
            self.update_prm()
            rate.sleep()


if __name__ == '__main__':
    try:
        prm = PRPM()
        prm.run()
    except rospy.ROSInterruptException:
        pass