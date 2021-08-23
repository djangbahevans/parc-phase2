#! /usr/bin/env python
import sys

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool


class TrafficDetector:
    def __init__(self) -> None:
        rospy.init_node("traffic_detector")

        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.image_cb)
        self.traffic_pub = rospy.Publisher(
            "/traffic_green", Bool, queue_size=0)

        self.bridge = CvBridge()

        rospy.spin()

    def image_cb(self, msg):
        # Capture frame-by-frame
        captured_frame = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding="bgr8")

        # First blur to reduce noise prior to color space conversion
        captured_frame_bgr = cv2.medianBlur(captured_frame, 3)

        # Convert to HSV color space
        captured_frame_hsv = cv2.cvtColor(
            captured_frame_bgr, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image, keep only the green pixels
        captured_frame_green = cv2.inRange(captured_frame_hsv, np.array(
            [40, 200, 200]), np.array([70, 255, 255]))

        # Second blur to reduce more noise, easier circle detection
        captured_frame_green = cv2.GaussianBlur(
            captured_frame_green, (5, 5), 2, 2)

        # Use the Hough transform to detect circles in the image
        circles = cv2.HoughCircles(captured_frame_green, cv2.HOUGH_GRADIENT, 1,
                                   captured_frame_green.shape[0] / 8, param1=20, param2=18, minRadius=5, maxRadius=25)

        # If we have extracted a circle, draw an outline
        # We only need to detect one circle here, since there will only be one reference object
        if circles is None:
            self.traffic_pub.publish(False)
        else:
            self.traffic_pub.publish(True)


if __name__ == "__main__":
    try:
        TrafficDetector()
    except rospy.ROSInterruptException:
        sys.exit()
