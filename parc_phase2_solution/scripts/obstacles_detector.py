#!/usr/bin/env python
from functools import reduce
import sys

import numpy as np
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from parc_phase2_solution.msg import Obstacle


class ObstacleChecker:
    def __init__(self) -> None:
        rospy.init_node("obstacle_detector")
        rospy.Subscriber("/scan", LaserScan, self.scan_cb)
        rospy.Subscriber("/odom", Odometry, self.odom_cb)

        self.obs_pub = rospy.Publisher("/obstacle", Obstacle, queue_size=10)

        self.odom = {}

        rospy.spin()

    def scan_cb(self, msg):
        """Handles LaserScan messages

        Args:
            msg (LaserScan): LaserScan data
        """
        angle_range = msg.angle_max - msg.angle_min
        msg_len = len(msg.ranges)

        self.obstacles = []
        for i in range(len(msg.ranges)):
            angle = i * angle_range/msg_len
            dist = msg.ranges[i]
            if dist != np.inf:
                y_dist = dist * np.sin(angle)
                x_dist = dist * np.cos(angle)
                self.obstacles.append((x_dist, y_dist))
        self.check_for_obstacles()

    def odom_cb(self, msg):
        """Handles odometry messages

        Args:
            msg (Odometry): Odometry data
        """
        position = msg.pose.pose.position
        rotation = msg.pose.pose.orientation
        (_, _, theta) = euler_from_quaternion(
            [rotation.x, rotation.y, rotation.z, rotation.w])
        self.odom["x"] = position.x
        self.odom["y"] = position.y
        self.odom["theta"] = theta
        self.odom_start = True

    def check_for_obstacles(self):
        """Checks for obstacles and sets self.obstacle to true if obstacle is found in the path of the robot.
        """
        obs = False
        obs_p = []
        for point in self.obstacles:
            if -0.15 <= point[1] <= 0.15:  # robot is 178mm wide
                # Obstacles should be less than 0.3 m away before being detected
                if 0 <= point[0] <= .2:
                    obs_p.append(point)
                    obs = True
        if obs:
            pos = self.determine_pos_of_obstacle(obs_p)
            data = Obstacle()
            data.x = pos[0]
            data.y = pos[1]
            data.obstacle = True
            self.obs_pub.publish(data)

    def determine_pos_of_obstacle(self, collisions):
        """Determines the position of the obstacles and returns the nearest point

        Returns:
            tuple[list[float, float], list[float, float]]: return[0] represents the nearest obstacle postion. return[1] is ideal offset point for robot to go to
        """
        if self.average_point(*collisions)[1] > 0:  # Obstacles on the left
            # return right most point
            true_obs_point = min(collisions, key=lambda x: x[1])
        else:  # Obstacles on the right
            # return left most point
            true_obs_point = max(collisions, key=lambda x: x[1])

        return (true_obs_point[0], true_obs_point[1])

    def average_point(self, *points):
        """Calculates the average of points

        Returns:
            tuple[float, float]: (x, y) representing the average of all the points
        """
        length = len(points)
        sum_x = reduce(lambda total, point: total + point[0], points, 0)
        sum_y = reduce(lambda total, point: total + point[1], points, 0)
        return (sum_x/length, sum_y/length)


if __name__ == "__main__":
    try:
        ObstacleChecker()
    except:
        sys.exit()
