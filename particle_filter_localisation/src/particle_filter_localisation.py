#!/usr/bin/env python3

import rospy
import math
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan
# import cv2 as cv # OpenCV2
import random
import numpy as np
import copy
import tf2_ros
from threading import Lock
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

from distance_transform import distance_transform
import time

from std_msgs.msg import Int32
import cv2



class Particle:
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def random_uniform(a, b):
    # Returns a random number with uniform distribution between "a" and "b"
    if b < a:
        rospy.logerror("The first argument must be less than the second argument when using the random_uniform method")
    return random.uniform(a, b)

def random_normal(stddev):
    # Returns a random number with normal distribution, 0 mean and a standard deviation of "stddev"
    return np.random.normal(0.0, stddev)

class ParticleFilter:
    def __init__(self):

        # Parameters
        # Default values will be used if no ROS parameter exists (see launch file)
        self.num_particles_ = rospy.get_param("~num_particles", 500) # Number of particles
        self.num_motion_updates_ = rospy.get_param("~num_motion_updates", 5) # Number of motion updates before a sensor update -- laser
        self.num_motion_updates2_ = rospy.get_param("~num_motion_updates", 2) # Number of motion updates before a sensor update -- terrain
        self.num_scan_rays_ = rospy.get_param("~num_scan_rays", 6) # (Approximate) number of scan rays to evaluate
        self.num_sensing_updates_ = rospy.get_param("~num_sensing_updates", 5) # Number of sensing updates before resampling
        self.motion_distance_noise_stddev_ = rospy.get_param("~motion_distance_noise_stddev", 0.01) # Standard deviation of distance noise for motion update
        self.motion_rotation_noise_stddev_ = rospy.get_param("~motion_rotation_noise_stddev", math.pi / 60.) # Standard deviation of rotation noise for motion update
        self.sensing_noise_stddev_ = rospy.get_param("~sensing_noise_stddev", 0.5) # Standard deviation of sensing noise
        self.magnetometer_noise_stddev_ = 0.349066 # 20 degrees # Standard deviation of magnetometer noise
        self.fraction_random_particles_ = rospy.get_param("~fraction_random_particles", 0.1) # when regenerated, this fraction of particles will be random
        self.clicked_point_std_dev_ = rospy.get_param("~clicked_point_std_dev", 0.5)

        self.use_terrain_ = rospy.get_param("~use_terrain")
        self.use_laser_ = rospy.get_param("~use_laser")
        self.use_compass_ = rospy.get_param("~use_compass")


        # Get the map via a ROS service call
        rospy.loginfo("Waiting for static_map service...")
        rospy.wait_for_service('static_map')
        get_map_service = rospy.ServiceProxy('static_map', GetMap)
        try:
            resp = get_map_service()
            self.map_ = resp.map
        except rospy.ServiceException as exc:
            rospy.logerror('Service did not process request: ' + str(exc))
            rospy.signal_shutdown()
        rospy.loginfo("Map received")

        # Convert occupancy grid into a numpy array
        self.map_image_ = np.reshape(self.map_.data, (self.map_.info.height, self.map_.info.width)).astype(np.int32)
        print(self.map_image_)

        # Limits of map in meters
        self.map_x_min_ = self.map_.info.origin.position.x
        self.map_x_max_ = self.map_.info.width * self.map_.info.resolution + self.map_.info.origin.position.x
        self.map_y_min_ = self.map_.info.origin.position.y
        self.map_y_max_ = self.map_.info.height * self.map_.info.resolution + self.map_.info.origin.position.y

        # Preprocess the distance transform for fast ray casting
        self.map_image_distance_transform_ = distance_transform(self.map_image_)

        # Read in the visual terrain map
        self.visual_terrain_map_ = VisualTerrainMap()

        # Variables
        self.particles_ = [] # Vector that holds the particles
        self.prev_odom_msg_ = None # Stores the previous odometry message to determine distance travelled
        self.estimated_pose_ = Pose() 
        self.compass_ = 0.
        self.compass_valid_ = False

        # Counters
        self.motion_update_count_ = 0 # Number of motion updates since last laser sensor update
        self.motion_update_count2_ = 0 # Number of motion updates since last terrain sensor update
        self.sensing_update_count_ = 0 # Number of sensing updates since resampling
        self.estimated_pose_valid_ = False # Don't use the estimated pose just after initialisation
        self.estimated_pose_theta_ = 0

        # Create a thread lock to avoid having different callbacks changing the particles_
        self.lock_ = Lock()
        self.lock_.acquire()

        # Marker for debug laser scan
        self.marker_ = Marker()
        self.marker_.header.frame_id = "odom"
        self.marker_.ns = "ns"
        self.marker_.id = 0
        self.marker_.type = Marker.LINE_LIST
        self.marker_.action = Marker.ADD
        self.marker_.pose.position.x = 0.0
        self.marker_.pose.position.y = 0.0
        self.marker_.pose.position.z = 0.0
        self.marker_.pose.orientation.x = 0.0
        self.marker_.pose.orientation.y = 0.0
        self.marker_.pose.orientation.z = 0.0
        self.marker_.pose.orientation.w = 1.0
        self.marker_.scale.x = 0.01
        self.marker_.scale.y = 0.1
        self.marker_.scale.z = 0.1
        self.marker_.color.a = 1.0
        self.marker_.color.r = 0.588235
        self.marker_.color.g = 0.0
        self.marker_.color.b = 0.0
        self.marker2_ = Marker()
        self.marker2_.header.frame_id = "odom"
        self.marker2_.ns = "ns2"
        self.marker2_.id = 1
        self.marker2_.type = Marker.CYLINDER
        self.marker2_.action = Marker.ADD
        self.marker2_.pose.position.x = 0.0
        self.marker2_.pose.position.y = 0.0
        self.marker2_.pose.position.z = 0.0
        self.marker2_.pose.orientation.x = 0.0
        self.marker2_.pose.orientation.y = 0.0
        self.marker2_.pose.orientation.z = 0.0
        self.marker2_.pose.orientation.w = 1.0
        self.marker2_.scale.x = 0.4
        self.marker2_.scale.y = 0.4
        self.marker2_.scale.z = 0.05
        self.marker2_.color.a = 1.0
        self.marker2_.color.r = 1.0
        self.marker2_.color.g = 1.0
        self.marker2_.color.b = 1.0
        self.marker3_ = Marker()
        self.marker3_.header.frame_id = "odom"
        self.marker3_.ns = "ns2"
        self.marker3_.id = 2
        self.marker3_.type = Marker.CYLINDER
        self.marker3_.action = Marker.ADD
        self.marker3_.pose.position.x = 0.0
        self.marker3_.pose.position.y = 0.0
        self.marker3_.pose.position.z = 0.0
        self.marker3_.pose.orientation.x = 0.0
        self.marker3_.pose.orientation.y = 0.0
        self.marker3_.pose.orientation.z = 0.0
        self.marker3_.pose.orientation.w = 1.0
        self.marker3_.scale.x = 0.45
        self.marker3_.scale.y = 0.45
        self.marker3_.scale.z = 0.02
        self.marker3_.color.a = 1.0
        self.marker3_.color.r = 0.0
        self.marker3_.color.g = 0.0
        self.marker3_.color.b = 0.0
        self.marker_pub_ = rospy.Publisher('marker', Marker, queue_size=100)
        
        # ROS Publishers
        self.particles_pub_ = rospy.Publisher('particles', PoseArray, queue_size=1)
        self.particles_seq_ = 0
        self.particles_pub_timer_ = rospy.Timer(rospy.Duration(0.1), self.publish_particles)

        self.estimated_pose_pub_ = rospy.Publisher('estimated_pose', PoseStamped, queue_size=1)
        self.estimated_pose_seq_ = 0
        self.estimated_pose_pub_timer_ = rospy.Timer(rospy.Duration(0.1), self.publish_estimated_pose)

        self.transform_broadcaster_ = tf2_ros.TransformBroadcaster()
        self.transform_seq_ = 0

        # ROS Subscribers
        self.odom_sub_ = rospy.Subscriber('odom', Odometry, self.odom_callback, queue_size=1) # Subscribes to wheel odometry
        self.scan_sub_ = rospy.Subscriber('base_scan', LaserScan, self.scan_callback, queue_size=1) # Subscribes to laser scan
        self.compass_sub_ = rospy.Subscriber('compass', Float32, self.compass_callback, queue_size=1) # Subscribes to compass
        self.terrain_sub_ = rospy.Subscriber('terrain_class', Int32, self.terrain_callback, queue_size=1) # Subscribes to terrain class observation
        self.clicked_point_sub_ = rospy.Subscriber('clicked_point', PointStamped, self.clicked_point_callback, queue_size=1) # Subscribes to clicked point from rviz

        # Initialise particles
        self.initialise_particles()

        self.lock_.release()

    def initialise_particles(self):
        # Create a set of random particles
        # "self.num_particles_" is the number of particles you will create

        # Clear the particles array
        self.particles_ = []

        # You want to initialise the particles in the "self.particles_" array
        # "random_uniform(a, b)" will give you a random value with uniform distribution between "a" and "b"
        # "self.map_x_min_", "self.map_x_max_", "self.map_y_min_", and "self.map_y_max_" give you the limits of the map
        # Orientation (theta) should be between 0 and 2*Pi

        ####################
        ## YOUR CODE HERE ##
        ## TASK 1         ##
        ####################

        self.particles_ = [Particle(random_uniform(self.map_x_min_, self.map_x_max_), random_uniform(self.map_y_min_, self.map_y_max_), 
                                    random_uniform(0, 2*math.pi), 1.0/self.num_particles_) for particle in range(self.num_particles_)]
        
        # Don't use the estimated pose just after initialisation
        self.estimated_pose_valid_ = False

        # Induce a sensing update
        self.motion_update_count_ = self.num_motion_updates_
        self.motion_update_count2_ = self.num_motion_updates2_

    def clicked_point_callback(self, clicked_point_msg):
        # This function is called when the "Publish Point" button in rviz is used to click on a point
        # reinitialise the particles, focussed around the clicked point in rviz
        # "self.num_particles_" is the number of particles you will create

        # Clear the particles array
        self.particles_ = []

        # Similar to initialise_particles(), except use a Gaussian distribution around the clicked point
        # The disk has standard deviation self.clicked_point_std_dev_
        # The clicked point message is of type geometry_msgs/PointStamped: 
        # https://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/PointStamped.html

        # Hint: use numpy.random.normal() for a Gaussian distribution for x and y
        # Hint: numpy has been imported as "np"
        # Hint: The orientation (theta) should be uniform between 0 and 2*pi

        ####################
        ## YOUR CODE HERE ##
        ## Task 3         ##
        ####################
        x_clicked = clicked_point_msg.point.x
        y_clicked = clicked_point_msg.point.y

        # intialise particles around the clicked point
        self.particles_ = [Particle(np.random.normal(x_clicked, self.clicked_point_std_dev_), np.random.normal(y_clicked, self.clicked_point_std_dev_), 
                                    np.random.uniform(0, 2*math.pi), 1.0/self.num_particles_) for particle_ in range(self.num_particles_)]

        # Don't use the estimated pose just after initialisation
        self.estimated_pose_valid_ = False

        # Induce a sensing update
        self.motion_update_count_ = self.num_motion_updates_
        self.motion_update_count2_ = self.num_motion_updates2_

        
    def normalise_weights(self):
        # Normalise the weights of the particles in "self.particles_"

        ####################
        ## YOUR CODE HERE ##
        ## Task 2         ##
        ####################
        # Normalise the weights of the particles in "self.particles_"
        weight_sum = sum(Particle.weight for Particle in self.particles_)
        if weight_sum > 0:  # To avoid division by zero and error in normalisation
            for particle in self.particles_:
                particle.weight = particle.weight / weight_sum
        else:
            # case where all weights are zero or particles are empty
            if len(self.particles_) > 0:
                equal_weight = 1.0 / len(self.particles_)
            for particle in self.particles_:
                particle.weight = equal_weight        
            else:
                pass


    def hit_scan(self, start_x, start_y, theta, max_range, draw=False):
        # Find the nearest obstacle from position start_x, start_y (in metres) in direction theta

        # Start point in occupancy grid coordinates
        start_point = [int(round((start_x - self.map_.info.origin.position.x) / self.map_.info.resolution)),
                             int(round((start_y - self.map_.info.origin.position.y) / self.map_.info.resolution))]

        # End point in real coordinates
        end_x = start_x + math.cos(theta) * max_range
        end_y = start_y + math.sin(theta) * max_range

        # End point in occupancy grid coordinates
        end_point = [int(round((end_x - self.map_.info.origin.position.x) / self.map_.info.resolution)),
                           int(round((end_y - self.map_.info.origin.position.y) / self.map_.info.resolution))]

        # Find the first "hit" along scan
        # (unoccupied is value 0, occupied is value 100)
        # hit = find_hit(self.map_image_, start_point, end_point, 50)
        hit = find_hit_df(self.map_image_distance_transform_, start_point, end_point)

        # Convert hit back to world coordinates
        hit_x = hit[0] * self.map_.info.resolution + self.map_.info.origin.position.x
        hit_y = hit[1] * self.map_.info.resolution + self.map_.info.origin.position.y

        # Add a debug visualisation marker
        if draw:
            point = Point(start_x, start_y, 0.)
            self.marker_.points.append(point)
            point = Point(hit_x, hit_y, 0.)
            self.marker_.points.append(point)

        # Get distance to hit
        return math.sqrt(math.pow(start_x - hit_x, 2) + math.pow(start_y - hit_y, 2))


    def estimate_pose(self):
        # Position of the estimated pose
        estimated_pose_x = 0.0
        estimated_pose_y = 0.0
        estimated_pose_theta = 0.0

        # Choose a method to estimate the pose from the particles in the "particles_" vector
        # Put the values into "estimated_pose_x", "estimated_pose_y", and "estimated_pose_theta"
        # If you just use the pose of the particle with the highest weight the maximum mark you can get for this part is 0.5

        ####################
        ## YOUR CODE HERE ##
        ## Task 6         ##
        ####################
        #intialise the variables
        weighted_sum_x = 0.0
        weighted_sum_y = 0.0
        total_weight = 0.0
        weight_sum_cos_theta = 0.0
        weight_sum_sin_theta = 0.0

        # loop through the particles
        
        for p in self.particles_:
            p_x = p.x
            p_y = p.y
            p_theta = p.theta
            p_weight = p.weight
            weighted_sum_x += p_x * p_weight
            weighted_sum_y += p_y * p_weight
            # re calc for theta
            weight_sum_cos_theta += np.cos(p_theta) * p_weight
            weight_sum_sin_theta += np.sin(p_theta) * p_weight
            total_weight += p_weight
            # re - normalise the weights in the particles if weight > 0 
            if p_weight > 0:
                p_weight = p_weight / total_weight
                # the x and y
                estimated_pose_x = weighted_sum_x / total_weight
                estimated_pose_y = weighted_sum_y / total_weight
                #for theta
                cos_theta_ = weight_sum_cos_theta / total_weight 
                sin_theta_ = weight_sum_sin_theta / total_weight
                estimated_pose_theta = np.arctan2(sin_theta_, cos_theta_)


        # Set the estimated pose message
        self.estimated_pose_.position.x = estimated_pose_x
        self.estimated_pose_.position.y = estimated_pose_y

        self.estimated_pose_.orientation.w = math.cos(estimated_pose_theta / 2.)
        self.estimated_pose_.orientation.z = math.sin(estimated_pose_theta / 2.)

        self.estimated_pose_theta_ = estimated_pose_theta
        self.estimated_pose_valid_ = True


    def resample_particles(self):
        # Resample the particles
        # Weights are expected to be normalised

        # Copy old particles
        old_particles = copy.deepcopy(self.particles_)
        self.particles_ = []

        # Keep a small fraction of random particles
        self.initialise_particles()
        if len(self.particles_) > 0:
            fraction_to_keep = self.fraction_random_particles_ # between 0 and 1
            number_to_keep = round(len(self.particles_)*fraction_to_keep)
            self.particles_ = self.particles_[0:number_to_keep]

        # Iterator for old_particles
        old_particles_i = 0

        # Find a new set of particles by randomly stepping through the old set, biased by their probabilities
        while len(self.particles_) < self.num_particles_:
            value = random_uniform(0.0, 1.0)
            sum = 0.0

            # Loop until a particle is found
            particle_found = False
            while not particle_found:

                # If the random value is between the sum and the sum + the weight of the particle
                if value > sum and value < (sum + old_particles[old_particles_i].weight):

                    # Add the particle to the "particles_" vector
                    self.particles_.append(copy.deepcopy(old_particles[old_particles_i]))

                    # Add jitter to the particle
                    self.particles_[-1].x = self.particles_[-1].x + random_normal(0.02)
                    self.particles_[-1].y = self.particles_[-1].y + random_normal(0.02)
                    self.particles_[-1].theta = wrap_angle(self.particles_[-1].theta + random_normal(math.pi / 30.))

                    # The particle may be out of the map, but that will be fixed by the motion update
                    
                    # Break out of the loop
                    particle_found = True

                # Add particle weight to sum and increment the iterator
                sum = sum + old_particles[old_particles_i].weight
                old_particles_i = old_particles_i + 1

                # If the iterator is past the vector, loop back to the beginning
                if old_particles_i >= len(old_particles):
                    old_particles_i = 0

        # Normalise the new particles
        self.normalise_weights()

        # Don't use the estimated pose just after resampling
        self.estimated_pose_valid_ = False

        # Induce a sensing update
        self.motion_update_count_ = self.num_motion_updates_ 
        self.motion_update_count2_ = self.num_motion_updates2_ 


    def publish_particles(self, event):
        self.lock_.acquire()
        pose_array = PoseArray()

        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = "odom"
        pose_array.header.seq = self.particles_seq_
        self.particles_seq_ = self.particles_seq_ + 1

        for p in self.particles_:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            pose.orientation.w = math.cos(p.theta / 2.)
            pose.orientation.z = math.sin(p.theta / 2.)

            pose_array.poses.append(pose)

        self.particles_pub_.publish(pose_array)
        self.lock_.release()

    def publish_estimated_pose(self, event):        
        if not self.estimated_pose_valid_:
            return

        self.lock_.acquire()

        # Publish the estimated pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "odom"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.seq = self.estimated_pose_seq_
        self.estimated_pose_seq_ = self.estimated_pose_seq_ + 1

        pose_stamped.pose = self.estimated_pose_

        self.estimated_pose_pub_.publish(pose_stamped)

        # Broadcast "odom" to "base_footprint" transform
        transform = TransformStamped()

        transform.header.frame_id = "odom"
        transform.header.stamp = rospy.Time.now()
        transform.header.seq = self.transform_seq_
        self.transform_seq_ = self.transform_seq_ + 1

        transform.child_frame_id = "base_footprint"

        transform.transform.translation.x = self.estimated_pose_.position.x
        transform.transform.translation.y = self.estimated_pose_.position.y

        transform.transform.rotation.w = self.estimated_pose_.orientation.w
        transform.transform.rotation.z = self.estimated_pose_.orientation.z

        # self.transform_broadcaster_.sendTransform(transform) # Commented by Graeme to avoid warning
        self.lock_.release()

    def odom_callback(self, odom_msg):
        # Receive an odometry message

        # Skip the first call since we are looking for movements
        if not self.prev_odom_msg_:
            self.prev_odom_msg_ = odom_msg
            return

        # Distance moved since the previous odometry message
        global_delta_x = odom_msg.pose.pose.position.x - self.prev_odom_msg_.pose.pose.position.x
        global_delta_y = odom_msg.pose.pose.position.y - self.prev_odom_msg_.pose.pose.position.y

        distance = math.sqrt(math.pow(global_delta_x, 2.) + math.pow(global_delta_y, 2.))

        # Previous robot orientation
        prev_theta = 2 * math.acos(self.prev_odom_msg_.pose.pose.orientation.w)

        if self.prev_odom_msg_.pose.pose.orientation.z < 0.:
            prev_theta = -prev_theta

        # Figure out if the direction is backward
        if (prev_theta < 0. and global_delta_y > 0.) or (prev_theta > 0. and global_delta_y < 0.):
            distance = -distance

        # Current orientation
        theta = 2 * math.acos(odom_msg.pose.pose.orientation.w)

        if odom_msg.pose.pose.orientation.z < 0.:
            theta = -theta

        # Rotation since the previous odometry message
        rotation = theta - prev_theta

        # Return if the robot hasn't moved
        if distance == 0 and rotation == 0:
            return

        self.lock_.acquire()

        # Motion update: add "distance" and "rotation" to each particle
        # You also need to add noise, which should be different for each particle
        # Use "random_normal()" with "self.motion_distance_noise_stddev_" and "self.motion_rotation_noise_stddev_" to get random values
        # You will probably need "math.cos()" and "math.sin()", and you should wrap theta with "wrap_angle()" too

        ####################
        ## YOUR CODE HERE ##
        ## Task 4         ##
        ####################
        
        for p in self.particles_: 
            noise_distance = random_normal(self.motion_distance_noise_stddev_) + distance
            noise_rotation = random_normal(self.motion_rotation_noise_stddev_) + rotation

            p.x += noise_distance * math.cos(p.theta)
            p.y += noise_distance * math.sin(p.theta)

            p.theta = wrap_angle(p.theta + noise_rotation)
        
        # Overwrite the previous odometry message
        self.prev_odom_msg_ = odom_msg

        # Delete any particles outside of the map
        old_particles = copy.deepcopy(self.particles_)
        self.particles_ = []

        for p in old_particles:
            if not(p.x < self.map_x_min_ or p.x > self.map_x_max_ or p.y < self.map_y_min_ or p.y > self.map_y_max_):
                # Keep it
                self.particles_.append(p)

        # Normalise particle weights because particles have been deleted
        self.normalise_weights()

        # If the estimated pose is valid move it too
        if self.estimated_pose_valid_:
            estimated_pose_theta = 2. * math.acos(self.estimated_pose_.orientation.w)

            if self.estimated_pose_.orientation.z < 0.:
                estimated_pose_theta = -estimated_pose_theta

            self.estimated_pose_.position.x += math.cos(estimated_pose_theta) * distance
            self.estimated_pose_.position.y += math.sin(estimated_pose_theta) * distance

            estimated_pose_theta = wrap_angle(estimated_pose_theta + rotation)

            self.estimated_pose_.orientation.w = math.cos(estimated_pose_theta / 2.)
            self.estimated_pose_.orientation.z = math.sin(estimated_pose_theta / 2.)

        # Increment the motion update counter
        self.motion_update_count_ = self.motion_update_count_ + 1
        self.motion_update_count2_ = self.motion_update_count2_ + 1

        self.lock_.release()

    def terrain_callback(self, terrain_msg):
        # Receive a visual terrain observation message

        # Visualisation for debugging
        if self.use_terrain_:
            if self.estimated_pose_valid_:

                self.lock_.acquire()
                col = self.visual_terrain_map_.colormap_[terrain_msg.data]
                x = self.estimated_pose_.position.x
                y = self.estimated_pose_.position.y
                self.lock_.release()

                self.marker2_.pose.position.x = x
                self.marker2_.pose.position.y = y
                self.marker2_.color.r = col[0] / 255.0
                self.marker2_.color.g = col[1] / 255.0
                self.marker2_.color.b = col[2] / 255.0

                self.marker3_.pose.position.x = x
                self.marker3_.pose.position.y = y
                

        self.marker_pub_.publish(self.marker2_)
        self.marker_pub_.publish(self.marker3_)

        # Only do a sensor update after num_motion_updates
        if self.motion_update_count2_ < self.num_motion_updates2_:
            return

        self.lock_.acquire()

        likelihood = 1.0

        if self.use_terrain_:

            # For each particle
            for p in self.particles_:

                # Compute the likelihood of making the received observation, 
                # assuming that this particle is at the correct location
                # Intuitively: If the received observation has a high likelihood of matching the 
                # observation at the location of this particle, then increase the weight of this particle

                ####################
                ## YOUR CODE HERE ##
                ## Task 5         ##
                ####################
                # likelihood = ??
                # p.weight = ??
                #retrieve the terrain at the location
                observed_terrain = terrain_msg.data
                # for ground truth
                p_x = p.x
                p_y = p.y
                actual_terrain = self.visual_terrain_map_.get_ground_truth(p_x, p_y)

                # for the likelihood
                likelihood = self.visual_terrain_map_.confusion_matrix[observed_terrain][actual_terrain]
                p.weight *= likelihood

                # Normalise particle weights
                self.normalise_weights()

                self.motion_update_count2_ = 0

            self.lock_.release()

        

    def scan_callback(self, scan_msg):
        # Receive a laser scan message

        # Only do a sensor update after num_motion_updates_
        if self.motion_update_count_ < self.num_motion_updates_:
            return

        self.lock_.acquire()

        if self.use_laser_:

            # Determine step size (the step may not result in the correct number of rays
            step = int(math.floor(float(len(scan_msg.ranges)) / self.num_scan_rays_))

            # Setup visualisation marker
            self.marker_.points = []

            start = time.time()

            # For each particle
            first_particle = True
            for p in self.particles_:
        
                # The likelihood of the particle is the product of the likelihood of each ray
                likelihood = 1.

                # Compare each scan ray
                for i in range(0, len(scan_msg.ranges), step): 
                    # The range value from the scan message
                    scan_range = scan_msg.ranges[i]

                    # The angle of the ray in the frame of the robot
                    local_angle = (scan_msg.angle_increment * i) + scan_msg.angle_min

                    # The angle of the ray in the map frame
                    global_angle = wrap_angle(p.theta + local_angle)

                    # The expected range value for the particle
                    particle_range = self.hit_scan(p.x, p.y, global_angle, 7.0, False) #first_particle) #scan_msg.range_max)

                    # Use "scan_range" and "particle_range" to get a likelihood
                    # Multiply the ray likelihood into the "likelihood" variable
                    # You will probably need "math.pi", math.sqrt()", "math.pow()", and "math.exp()"

                    ####################
                    ## YOUR CODE HERE ##
                    ## Task 7         ##
                    ####################
                    # likelihood = ??
                    ## writing the equation
                    ray_likelihood = (1/(math.sqrt(2*math.pi*math.pow(self.sensing_noise_stddev_,2))))*math.exp(-(math.pow(particle_range - scan_range, 2)/(2*math.pow(self.sensing_noise_stddev_, 2))))
                    likelihood *= ray_likelihood 
             # Update the particle weight with the likelihood
                p.weight *= likelihood

                first_particle = False

            end = time.time()

            print('ray casting time', round(end - start, 4))

        # Normalise particle weights
        self.normalise_weights()

        # Estimate the pose of the robot
        self.estimate_pose()

        self.motion_update_count_ = 0

        self.sensing_update_count_ = self.sensing_update_count_ + 1

        if self.sensing_update_count_ > self.num_sensing_updates_:
            self.resample_particles()
            self.sensing_update_count_ = 0

        # Draw the debug marker
        if self.use_laser_:
            if self.estimated_pose_valid_:

                theta = self.estimated_pose_theta_
                x = self.estimated_pose_.position.x
                y = self.estimated_pose_.position.y

                for i in range(0, len(scan_msg.ranges), step): 
                    # The range value from the scan message
                    scan_range = scan_msg.ranges[i]

                    # The angle of the ray in the frame of the robot
                    local_angle = (scan_msg.angle_increment * i) + scan_msg.angle_min

                    # The angle of the ray in the map frame
                    global_angle = wrap_angle(theta + local_angle)

                    # Plot it
                    self.hit_scan(x, y, global_angle, 7.0, True)

        self.lock_.release()

        # Publish debug marker
        self.marker_pub_.publish(self.marker_)


    def compass_fusion(self):

        # Return if haven't received a compass measurement
        if not self.compass_valid_:
            return

        if not self.use_compass_:
            return

        # Extension task: Can you think of a way to incorporate the magnetometer measurements into the particles?
        #
        # Compass measurements are in self.compass_ and are in the same orientation as particle.theta 
        # Compass measurements have Gaussian noise with standard deviation self.magnetometer_noise_stddev_ 

        ####################
        ## YOUR CODE HERE ##
        ## Task 8         ##
        ####################
        compass_measurement = self.compass_

        for particle in self.particles_:
        #  as compass gives the heading use the heading to update the particle's weight
        #  calc heading error 
        #  then weight the particle 
         # Calculate the difference between the compass measurement and the particle's heading
         heading_error = wrap_angle(compass_measurement - particle.theta)

         # Calculate the likelihood of this particle based on the heading error using Gaussian distribution
         heading_likelihood = (1 / (math.sqrt(2 * math.pi* math.pow(self.magnetometer_noise_stddev_,2)))) *math.exp(- (math.pow(heading_error, 2) / (2 * math.pow(self.magnetometer_noise_stddev_, 2))))

         # Update the particle's weight with this likelihood
        particle.weight *= heading_likelihood
        self.normalise_weights()
        self.estimate_pose()

        
    def compass_callback(self, compass_msg):

        self.lock_.acquire()

        self.compass_ = compass_msg.data
        self.compass_valid_ = True

        self.compass_fusion()

        self.lock_.release()


def find_hit(img, p1, p2, threshold):
    # Draws a line from p1 to p2
    # Stops at the first pixel that is a "hit", i.e. above the threshold
    # Returns the pixel coordinates for the first hit

    # Extract the vector
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    step = 3.0 # pixels

    dx = x2 - x1
    dy = y2 - y1
    l = math.sqrt(dx**2. + dy**2.)
    dx = step * dx / l
    dy = step * dy / l

    max_steps = int(l / step)

    for i in range(max_steps):

        # Get the next pixel
        x = int(round(x1 + dx*i))
        y = int(round(y1 + dy*i))

        # Check if it's outside of the image
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return [x, y] #p2

        # Check for "hit"
        if img[y, x] >= threshold:
            return [x, y]

    # No hits found
    return p2

def find_hit_df(img, p1, p2):
    # Draws a line from p1 to p2
    # Stops at the first pixel that is a "hit", i.e. above the threshold
    # Returns the pixel coordinates for the first hit
    #
    # similar to find_hit but uses distance transform image to speed things up

    # Extract the vector
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    dx = x2 - x1
    dy = y2 - y1
    l = math.sqrt(dx**2. + dy**2.)
    dx = dx / l
    dy = dy / l

    step = 1.0 # pixels
    min_step = 5 # pixels -- too large risks jumping over obstacles

    max_steps = int(l)

    dist = 0

    while dist < max_steps:

        # Get the next pixel
        x = int(round(x1 + dx*dist))
        y = int(round(y1 + dy*dist))

        # Check if it's outside of the image
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return [x, y] #p2

        current_distance = img[y, x] 

        # Check for "hit"
        if current_distance <= 0:
            return [x, y]

        # Otherwise, adjust the step size according to the distance transform
        step = current_distance-1.
        if step < min_step:
            step = min_step

        # Move along the ray
        dist += step

    # No hits found
    return p2


class VisualTerrainMap:

    confusion_matrix = np.array(
        [[0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
         [0.02, 0.90, 0.02, 0.02, 0.02, 0.02],
         [0.02, 0.02, 0.90, 0.02, 0.02, 0.02],
         [0.02, 0.02, 0.02, 0.90, 0.02, 0.02],
         [0.02, 0.02, 0.02, 0.02, 0.90, 0.02],
         [0.02, 0.02, 0.02, 0.02, 0.02, 0.90]])

    def __init__(self):
        
        # filenames
        self.filename_class_map_ = rospy.get_param("~filename_class_map")
        self.filename_class_colour_map_ = rospy.get_param("~filename_class_colour_map")

        # read in the images
        self.class_map_ = cv2.imread(self.filename_class_map_)
        self.class_colour_map_ = cv2.imread(self.filename_class_colour_map_)

        self.class_map_ = self.class_map_[:,:,0]

        # figure out the transformation
        self.resolution_ = rospy.get_param("~resolution")
        self.origin_ = rospy.get_param("~origin")

        self.shape_ = self.class_map_.shape

        # Reverse engineer the colour scheme for visualisation
        self.colormap_ = []

        for idx in range(len(self.confusion_matrix[0])):

            found = False

            # Find the idx in the image
            for i in range(self.shape_[0]):
                for j in range(self.shape_[1]):
                    if self.class_map_[i, j] == idx:
                        found = True
                        break
                if found == True:
                    break

            # If it exists, extract the colour from the coloured map
            if found:
            
                # Store it
                col = self.class_colour_map_[i,j,:]
                col = np.flip(col)
                self.colormap_.append(col)

            else:
                # Add arbitrary colour
                self.colormap_.append([0, 0, 0])



    def get_ground_truth(self, x, y):

        # Transform to pixel coordinates
        j = round((x - self.origin_[0]) / self.resolution_)
        i = round(self.shape_[0]-1 - (y - self.origin_[1]) / self.resolution_)

        # Query map
        if i < 0 or j < 0 or i >= self.shape_[0] or j >= self.shape_[1]:
            # out of range, return arbitrary value
            ground_truth_class = 0
        else:
            ground_truth_class = self.class_map_[i, j]

        return ground_truth_class


if __name__ == '__main__':
    # Create the ROS node
    rospy.init_node('particle_filter_localisation')

    # Create the particle filter
    ParticleFilter()

    # Loop forever while processing callbacks
    rospy.spin()