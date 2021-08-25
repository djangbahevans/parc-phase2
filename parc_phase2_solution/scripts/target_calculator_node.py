#!/usr/bin/env python
from functools import reduce
import sys

import matplotlib.pyplot as plt
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from laser_line_extraction.msg import LineSegmentList
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from tf.listener import TransformListener
from tf.transformations import euler_from_quaternion
from parc_phase2_solution.msg import Obstacle

try:
    WALL_OFFSET = float(sys.argv[1])
except IndexError:
    rospy.logerr("usage: rosrun target_calculator_node.py <wall_offset>")
except ValueError as e:
    rospy.logfatal(str(e))
    rospy.signal_shutdown("Fatal error")

DISTANCE_AWAY = 1


class TargetCalculator:
    def __init__(self):
        rospy.init_node("target_calculator")

        self.listener = TransformListener()
        self.listener.waitForTransform(
            "odom", "base_footprint", rospy.Time(), rospy.Duration(20), rospy.Duration(60))

        rospy.Subscriber("/line_segments", LineSegmentList, self.line_cb)
        rospy.Subscriber("/odom", Odometry, self.odom_cb)
        rospy.Subscriber(
            "/obstacle", Obstacle, self.obstacle_cb)

        self.target_pub = rospy.Publisher(
            "/lane_target", Float64MultiArray, queue_size=100)

        self.odom = {"x": 0, "y": 0, "theta": 0}

        self.line = None
        self.obstacle = False

        rospy.timer.sleep(1)
        # while not rospy.is_shutdown():
        #     plt.cla()
        #     plt.axis([-3, 3, -4, 4])
        #     plt.scatter([self.start[0], self.x, self.odom["x"], self.new_point[0], self.off_x, self.end[0]], [
        #                 self.start[1], self.y, self.odom["y"], self.new_point[1], self.off_y, self.end[1]])
        #     plt.annotate("s", (self.start[0], self.start[1]))
        #     plt.annotate("ri", (self.x, self.y))
        #     plt.annotate("r", (self.odom["x"], self.odom["y"]))
        #     plt.annotate("oi", (self.new_point[0], self.new_point[1]))
        #     plt.annotate("o", (self.off_x, self.off_y))
        #     plt.annotate("e", (self.end[0], self.end[1]))
        #     plt.draw()
        #     plt.pause(0.0001)

        rospy.spin()

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

    def obstacle_cb(self, msg):
        self.obstacle = msg.obstacle

    def line_cb(self, msg):
        line_segs = msg.line_segments
        if len(line_segs) == 0 or self.obstacle:
            if self.line is None:
                return
            line = self.line
        elif len(line_segs) >= 1:
            # Choose suitable line
            line_segs.sort(key=lambda line: abs(np.arctan2(
                line.end[1] - line.start[1], line.end[0] - line.start[0])))
            self.line = line = line_segs[0]
            # Convert line to global frame
            self.start = self.transform_frame(
                line.start, from_='base_footprint', to='odom')
            self.end = self.transform_frame(
                line.end, from_='base_footprint', to='odom')

        # Find slope and intercept of line
        m = self.calculate_slope(self.start, self.end)
        c = self.calculate_intercept(self.start, self.end)
        # Find slope and intercept of perpendicular line
        perp_slope = -1/m
        perp_intercept = self.odom["y"] - (perp_slope * self.odom["x"])
        # Find point on wall perpendicular to current position
        b = self.odom["y"]
        a = self.odom["x"]
        # Robot intersect on wall
        self.x = (b*m + a - c*m)/(m**2 + 1)
        self.y = (m*(b*m + a) + c)/(m**2 + 1)
        # Finding point on wall, DISTANCE_AWAY away from current position
        v = np.array([self.x - self.end[0], self.y - self.end[1]])
        u = v/np.linalg.norm(v)
        # self.new_point = (-DISTANCE_AWAY * u) + np.array([self.x, self.y])
        self.new_point = self.furthest_ahead((-DISTANCE_AWAY * u) + np.array(
            [self.x, self.y]), (DISTANCE_AWAY * u) + np.array([self.x, self.y]))
        # Find offset of new point
        # Slope is perp_slope
        p_s = []
        self.off_x = self.new_point[0] + \
            np.sqrt(WALL_OFFSET**2/(1 + 1/(m**2)))
        self.off_y = perp_slope * \
            (self.off_x - self.new_point[0]) + self.new_point[1]
        p_s.append((self.off_x, self.off_y))
        self.off_x = self.new_point[0] - \
            np.sqrt(WALL_OFFSET**2/(1 + 1/(m**2)))
        self.off_y = perp_slope * \
            (self.off_x - self.new_point[0]) + self.new_point[1]
        p_s.append((self.off_x, self.off_y))
        self.target = self.closest_point(*p_s)
        (self.off_x, self.off_y) = self.target

        data = Float64MultiArray()
        data.data = self.target
        self.target_pub.publish(data)

    def furthest_ahead(self, p1, p2):
        p1_c = self.transform_frame(p1, from_="odom", to="base_footprint")
        p2_c = self.transform_frame(p2, from_="odom", to="base_footprint")
        if p1_c[0] > p2_c[0]:
            return p1
        else:
            return p2

    def closest_point(self, *args):
        x = self.odom["x"]
        y = self.odom["y"]
        min_dist = None
        for point in args:
            arr = np.array([point[0], point[1]]) - np.array([x, y])
            dist = np.linalg.norm(arr)
            if min_dist is None or dist < min_dist:
                min_dist = dist
                p = point

        return p

    def calculate_slope(self, start, end):
        return (start[1] - end[1])/(start[0] - end[0])

    def calculate_intercept(self, start, end):
        slope = self.calculate_slope(start, end)
        return start[1] - (slope * start[0])

    def transform_frame(self, p, from_, to):
        """Converts a point from robot frame (base_link) to the global frame (odom)

        Args:
            p (tuple[float, float]): Point in robot frame to convert

        Returns:
            list[float, float]: Converted point in global frame
        """
        point = PointStamped()
        point.header.frame_id = from_
        point.header.stamp = rospy.Time(0)
        point.point.x = p[0]
        point.point.y = p[1]
        point.point.z = 0.0
        transformed = self.listener.transformPoint(to, point)
        return (transformed.point.x, transformed.point.y)


if __name__ == "__main__":
    try:
        TargetCalculator()
    except rospy.ROSInterruptException:
        sys.exit()
