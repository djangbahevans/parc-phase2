#!/usr/bin/env python
import sys
from functools import reduce
from time import time
import actionlib
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from parc_phase2_solution.msg import Obstacle
from std_msgs.msg import Bool, Float64MultiArray
from tf.listener import TransformListener
from tf.transformations import euler_from_quaternion

from Graph import Graph, Vertex
from PID import PID

try:
    goal_x = float(sys.argv[1])
    goal_y = float(sys.argv[2])
    rospy.loginfo("Moving to {0}".format((goal_x, goal_y)))
except IndexError:
    rospy.logerr("usage: rosrun task_solution.py <goal_x> <goal_y>")
except ValueError as e:
    rospy.logfatal(str(e))
    rospy.signal_shutdown("Fatal error")


class TaskSolution:
    def __init__(self):
        rospy.init_node("task_solution")
        self.listener = TransformListener()
        self.listener.waitForTransform(
            "odom", "base_link", rospy.Time(), rospy.Duration(20))

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.bridge = CvBridge()
        self.obstacle = False
        self.odom = {}
        self.stored_odom = {}
        self.target = (0, 0)
        self.odom_start = False
        self.img_start = False
        self.target_start = False
        self.final = False
        self.traffic_green = False
        self.time_since_last = np.inf

        self.graph = self.create_map()
        self.angle_pid = PID(Kp=1, Ki=0, Kd=0, setpoint=.22,
                             output_limits=(-.5, .5))
        self.speed_pid = PID(Kp=-.2, Ki=0, Kd=0, setpoint=0,
                             output_limits=(-.22, .22))

        self.odom_sub = rospy.Subscriber(
            "/odom", Odometry, callback=self.odom_cb)
        self.target_sub = rospy.Subscriber(
            "/lane_target", Float64MultiArray, self.target_cb)
        self.obstacle_sub = rospy.Subscriber(
            "/obstacle", Obstacle, self.obstacle_cb)

        self.vel_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.obs_pub = rospy.Publisher("/obstacle", Obstacle, queue_size=10)

        while not (self.odom_start and self.target_start):
            if rospy.is_shutdown():
                sys.exit()

        self.main()
        # rospy.spin()

    def main(self):
        """Main loop of robot
        """
        # self.go_to_goal(self.p)
        v: Vertex = self.graph.get_vertex("J")
        first_point = self.find_first_point()
        (goal_node, _) = self.nearest_node(v.coordinates)
        path = self.find_shortest_path(
            first_point.get_id(), goal_node.get_id())
        rospy.loginfo("Using path {0}".format(path))
        rospy.loginfo("Moving to {0}".format(first_point.get_id()))
        self.go_to_intersection(first_point.coordinates, "lane")

        for i in range(1, len(path)):
            if i == len(path) - 1:
                self.final = True
            p = path[i]
            v: Vertex = self.graph.get_vertex(p)
            how = v.adjacent[self.graph.get_vertex(path[i-1])][1]
            h = np.arctan2(
                (v.coordinates[1] - self.odom["y"]), (v.coordinates[0] - self.odom["x"]))
            self.turn_to_heading(h)
            self.stop()
            rospy.loginfo("Moving from {0} to {1} using {2}".format(
                path[i-1], p, how))
            self.go_to_intersection(v.coordinates, how)

        # self.cluttered_navigation((goal_x, goal_y))
        rospy.logwarn("At destination")

    def go_to_intersection(self, p, how="lane"):
        """Knows how to navigate to any intersection at point p

        Args:
            p (tuple[float, float]): Coordinates of the intersection
        """
        self.store_current_heading()
        while not self.at_coordinates(p, radius=0.5, scheme="both"):
            if rospy.is_shutdown():
                sys.exit()
            if how == "lane":
                if not self.obstacle:
                    self.keep_going(kind="lane", speed=.1)
                else:
                    self.stop()
                    self.store_current_heading()
                    obs_offset_local = self.determine_pos_of_obstacle()
                    obs_offset_global = self.transform_frame(
                        obs_offset_local, from_="base_link", to="odom")
                    true_obs_global = self.transform_frame(
                        self.obs_point, from_="base_link", to="odom")
                    rospy.logwarn("Obstacle at {0}".format(true_obs_global))

                    self.go_to_goal(obs_offset_global, speed=.05, distance=.01)

                    self.turn_to_goal(self.target)
                    while not self.obstacle_behind(obs_offset_global, offset=0.2):
                        if rospy.is_shutdown():
                            sys.exit()
                        # self.turn_to_heading(self.current_heading, speed=0)
                        self.keep_going(kind="lane", speed=.05)

                    self.reset_obstacles()
            if how == "crossing":
                self.stop()
                self.turn_to_goal(p)
                self.crossing(p)
            if how == "cluttered":
                self.cluttered_navigation(p)
                break
        else:
            self.go_to_goal(p, distance=0.1)

    def cluttered_navigation(self, destination):
        self.client.wait_for_server()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "odom"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = destination[0]
        goal.target_pose.pose.position.y = destination[1]
        goal.target_pose.pose.orientation.w = 0.1
        self.client.send_goal(goal, feedback_cb=self.feedback_cb)
        wait = self.client.wait_for_result()
        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")
        else:
            self.client.get_result()

        # self.stop()

    def crossing(self, destination):
        traffic_sub = rospy.Subscriber(
            "/traffic_green", Bool, self.traffic_cb)  # subscribe to traffic state
        while not self.traffic_green:
            if rospy.is_shutdown():
                sys.exit()
        traffic_sub.unregister()  # stop subscribing as soon as green detected
        self.go_to_goal(destination, distance=.1)

    def find_first_point(self):
        """Find nodes in front of bot, then find the closest one.

        Returns:
            Vertex: The closest node to the robot
        """
        nodes = []
        for node in self.graph:
            # p = self.global_to_robot_frame(node.coordinates)
            p = self.transform_frame(
                node.coordinates, from_="odom", to="base_link")
            if p[0] >= 2:  # Ignore current node if there
                nodes.append(node)

        (node, _) = self.nearest_node((self.odom["x"], self.odom["y"]), nodes)
        return node

    def reset_obstacles(self):
        self.obstacle = False
        self.time_since_last = time()
        data = Obstacle()
        data.obstacle = False
        self.obs_pub.publish(data)

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

    def stop(self):
        """Stops the robot
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0
        cmd_vel.angular.z = 0
        self.vel_cmd.publish(cmd_vel)

    def obstacle_behind(self, point, offset=0):
        """Checks if the robot is in front of an obstacle point

        Args:
            point (tuple[float, float]): The point to check for
            offset (int, optional): How far behind the point should be for it to be detected. Defaults to 0.

        Returns:
            boolean: True if point is behind the robot
        """
        p = self.transform_frame(point, from_="odom", to="base_link")
        # p = self.global_to_robot_frame(point)
        if p[0] >= (0 - offset):
            return False
        else:
            return True

    def turn_to_heading(self, heading, speed=0):
        """Returns the robot to the set heading

        Args:
            heading (float): The heading to turn to
            speed (float, optional): The forward speed of the robot. Defaults to 0.
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = speed
        while abs(self.odom["theta"] - heading) > 0.5:
            ang_err = heading - self.odom["theta"]
            if ang_err >= np.pi:
                ang_err -= 2*np.pi
            elif ang_err <= -np.pi:
                ang_err += 2*np.pi
            cmd_vel.angular.z = np.clip(.5 * (ang_err), -2.84, 2.84)
            if rospy.is_shutdown():
                sys.exit()
            self.vel_cmd.publish(cmd_vel)

    def go_to_goal_no_hang(self, point, distance=0.1, dist_ctrl=True, speed=.1):
        cmd_vel = Twist()

        del_x = point[0] - self.odom["x"]
        del_y = point[1] - self.odom["y"]
        norm = np.sqrt(del_x**2 + del_y**2)
        d_theta = np.arctan2(del_y, del_x)

        if norm < distance:
            self.stop()
            return

        if dist_ctrl:
            cmd_vel.linear.x = self.speed_pid(norm)
        else:
            cmd_vel.linear.x = speed
        ang_err = d_theta - self.odom["theta"]
        if ang_err >= np.pi:
            ang_err -= 2*np.pi
        elif ang_err <= -np.pi:
            ang_err += 2*np.pi
        cmd_vel.angular.z = np.clip(.5 * (ang_err), -2.84, 2.84)
        self.vel_cmd.publish(cmd_vel)

    def turn_to_goal(self, point):
        cmd_vel = Twist()
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            del_x = point[0] - self.odom["x"]
            del_y = point[1] - self.odom["y"]
            d_theta = np.arctan2(del_y, del_x)
            ang_err = d_theta - self.odom["theta"]
            cmd_vel.linear.x = 0
            if ang_err >= np.pi:
                ang_err -= 2*np.pi
            elif ang_err <= -np.pi:
                ang_err += 2*np.pi
            cmd_vel.angular.z = np.clip(.5 * (ang_err), -2.84, 2.84)
            if ang_err <= 0.1:
                self.stop()
                break
            self.vel_cmd.publish(cmd_vel)
            rate.sleep()

    def go_to_goal(self, point, distance=0.1, dist_ctrl=True, speed=.1):
        """Navigates the robot straight to a point. Should only be used for short distances where there are no obstacles.

        Args:
            point (tuple[float, float]): The point to navigate to.
            distance (float, optional): The distance to assume destination reached. Defaults to 0.5.
        """
        cmd_vel = Twist()
        rate = rospy.Rate(100)

        while not rospy.is_shutdown():
            del_x = point[0] - self.odom["x"]
            del_y = point[1] - self.odom["y"]
            norm = np.sqrt(del_x**2 + del_y**2)
            d_theta = np.arctan2(del_y, del_x)
            if dist_ctrl:
                cmd_vel.linear.x = self.speed_pid(norm)
            else:
                cmd_vel.linear.x = speed
            ang_err = d_theta - self.odom["theta"]
            if ang_err >= np.pi:
                ang_err -= 2*np.pi
            elif ang_err <= -np.pi:
                ang_err += 2*np.pi
            cmd_vel.angular.z = np.clip(.5 * (ang_err), -2.84, 2.84)
            if norm < distance:
                self.stop()
                break
            self.vel_cmd.publish(cmd_vel)
            rate.sleep()

    def at_coordinates(self, coordinate, scheme="y", radius=0.01):
        """Checks if robot is at or near a particular coordinate.

        Args:
            coordinate (tuple[float, float]): The coordinate to check for.
            scheme (str, optional): Options are "x", "y" and "both". Used to determine how to calculate distance. "x" and "y" takes distance from global x and y axis respectively only into account, "both" calculates the norm from both axis. Defaults to "y".
            radius (float, optional): The radius distance from the to assume convergence. Defaults to 0.01.

        Returns:
            bool: Returns True if robot is within a specified distance from the point.
        """
        # Check if we have desired y
        if scheme == "y":
            return True if abs(self.odom["y"] - coordinate[1]) <= radius else False
        elif scheme == "x":
            return True if abs(self.odom["x"] - coordinate[0]) <= radius else False
        elif scheme == "both":
            return True if np.sqrt((self.odom["x"] - coordinate[0])**2 + (
                self.odom["y"] - coordinate[1])**2) <= radius else False

    def determine_pos_of_obstacle(self):
        """Determines the position of the obstacles and returns the nearest point

        Returns:
            tuple[list[float, float], list[float, float]]: return[0] represents the nearest obstacle postion. return[1] is ideal offset point for robot to go to
        """
        if self.obs_point[1] > 0:  # Obstacles on the left
            print("Obstacle on the left")
            obs_offset = [self.obs_point[0], self.obs_point[1] - .2]
        else:  # Obstacles on the right
            print("Obstacle on the right")
            obs_offset = [self.obs_point[0], self.obs_point[1] + .2]

        return (obs_offset[0], obs_offset[1])

    def store_current_heading(self):
        """Saves current heading for later retrieval
        """
        self.current_heading = self.odom["theta"]

    def keep_going(self, speed=0.1, kind="lane"):
        """Follows the lane based on camera data

        Args:
            speed (float, optional): The forward speed of the robot. Defaults to 0.5.
            kind (str, optional): The type of motion to use. Options are "lane" and "forward". Defaults to "lane".
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = speed
        if kind == "forward":
            cmd_vel.angular.z = 0
            self.vel_cmd.publish(cmd_vel)
        elif kind == "lane":
            self.go_to_goal_no_hang(self.target, dist_ctrl=False, speed=speed)

    def traffic_cb(self, msg):
        self.traffic_green = msg.data

    def obstacle_cb(self, msg):
        if self.obstacle:
            return

        if abs(time() - self.time_since_last) <= 1:
            return

        self.obstacle = msg.obstacle
        self.obs_point = (msg.x, msg.y)

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

    def target_cb(self, msg):
        self.target = msg.data
        self.target_start = True

    def feedback_cb(self, msg):
        position_x = msg.base_position.pose.position.x
        position_y = msg.base_position.pose.position.y
        if self.distance((position_x, position_y), (goal_x, goal_y)) < 0.18:  # Close enough
            # rospy.logwarn("At a distance of {0}".format(
            #     self.distance((position_x, position_y), (goal_x, goal_y))))
            self.client.cancel_all_goals()
            self.turn_to_goal((goal_x, goal_y))

    def average_point(self, *points):
        """Calculates the average of points

        Returns:
            tuple[float, float]: (x, y) representing the average of all the points
        """
        length = len(points)
        sum_x = reduce(lambda total, point: total + point[0], points, 0)
        sum_y = reduce(lambda total, point: total + point[1], points, 0)
        return (sum_x/length, sum_y/length)

    def find_shortest_path(self, start, end, path=[]):
        """Finds the shortest path between two nodes on the graph

        Args:
            start (str): The starting position or node
            end (str): The destination position or node
            path (list[str], optional): A list of previous path chosen. Defaults to [].

        Returns:
            list[str]: A list of nodes to move through to get to the desired node
        """
        path = path + [start]
        if start == end:
            return path
        if not start in self.graph.vert_dict:
            return None
        shortest = None
        for node in self.graph.vert_dict[start].adjacent:
            if node.get_id() not in path:
                newpath = self.find_shortest_path(node.get_id(), end, path)
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def nearest_node(self, p, nodes=[]):
        """Takes a point and returns the node closest to that point

        Args:
            p (tuple[float, float]): The point
            nodes (list[Vertex], optional): A list of vertices to scan. If not provided, the entire graph is scanned. Defaults to []

        Returns:
            str: The node as a string
        """
        search_space = self.graph if len(nodes) == 0 else nodes
        prev_dist = np.inf
        for node in search_space:
            dist = self.distance(p, node.coordinates)
            if dist < prev_dist:
                nearest_node = node
                prev_dist = dist

        return nearest_node, prev_dist

    def distance(self, p1, p2):
        """Calculates the distance between two points

        Args:
            p1 (tuple[float, float]): The first point
            p2 (tuple[float, float]): The second point

        Returns:
            float: The scalar distance between the two points
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def create_map(self):
        """Creates the graph intersection map of the PARC world

        Returns:
            Graph: The graph representation of the PARC world
        """
        g = Graph()
        g.add_vertex("A", (-1.70, -3.78))
        g.add_vertex("B", (1.70, -3.82))
        g.add_vertex("C", (-1.70, -0.72))
        g.add_vertex("D", (1.70, -0.72))
        g.add_vertex("E", (-1.70, 0.64))
        g.add_vertex("F", (1.70, 0.60))
        g.add_vertex("G", (-1.70, 3.67))
        g.add_vertex("H", (1.70, 3.70))
        g.add_vertex("I", (-1.60, 2.16))
        g.add_vertex("J", (goal_x, goal_y))

        g.add_edge("A", "B", "lane")
        g.add_edge("A", "C", "lane")

        g.add_edge("B", "D", "lane")

        g.add_edge("C", "D", "lane")
        g.add_edge("C", "E", "crossing")

        g.add_edge("D", "F", "crossing")

        g.add_edge("E", "F", "lane")
        g.add_edge("E", "G", "lane")

        g.add_edge("F", "H", "lane")

        g.add_edge("G", "H", "lane")
        g.add_edge("G", "I", "lane")

        g.add_edge("I", "E", "lane")
        g.add_edge("I", "J", "cluttered")

        return g


if __name__ == "__main__":
    try:
        TaskSolution()
    except rospy.ROSInterruptException:
        sys.exit()
