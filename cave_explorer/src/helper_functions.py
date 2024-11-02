import math
from geometry_msgs.msg import Pose


def wrap_angle(angle):
    # Function to wrap an angle between 0 and 2*Pi
    while angle < 0.0:
        angle = angle + 2 * math.pi
    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle


def pose2d_to_pose(pose_2d):
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


def compute_distance_between_points(point1, point2):
    return math.hypot((point1[0] - point2[0]), (point1[1] - point2[1]))