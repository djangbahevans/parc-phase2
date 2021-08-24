# Team Asimovs: PARC Engineers League  

## Introduction

The 2021 PARC Engineers League challenges the various teams to build a delivery robot that can pick up a package, and deliver it to a destination autonomously, while obeying traffic rules and avoid collision with nearby objects. Deliverty robots are seeing more use world wide, especially during the coronavirus pandemic, where they could deliver products autonomously. Delivery robots could be useful in rural cocoa farming communities, where they can autonomously deliver plucked cocoa pods to the processing points, removing the need to use people, mostly children, for this purposes.

**Team Country:** Ghana

**Team Member Names:**

* Evans Djangbah (Team Leader)
* Isaac Atia

## Dependencies

**Packages needed are:**

* `ros_control`: ROS packages including controller interfaces, controller managers, transmissions, etc.
  * `$ sudo apt-get install ros-melodic-ros-control ros-melodic-ros-controllers`

* `opencv`: Open-source computer vision library.
  * `$ sudo apt-get install python-opencv`

* `numpy`: A python package for scientific computing.
  * `$ sudo apt-get install python-numpy`

* `cv_bridge`: CvBridge is a ROS library that provides an interface between ROS and OpenCV.
  * `$ sudo apt-get install ros-melodic-cv-bridge`

* `navigation`: ROS navigation provides packages that can be used to plan paths and navigate around cluttered environments.
  * `$ sudo apt-get install ros-melodic-navigation`

* `tf`: tf is a package that lets the user keep track of multiple coordinate frames over time.
  * `$ sudo apt-get install ros-melodic-tf`
  * `$ sudo apt-get install ros-melodic-tf2`

* `laser_line_extraction`: laser_line_extraction is used to extract straight lines from laser scans. Ideal for identifying walls and other objects.
  * ` git clone https://github.com/kam3k/laser_line_extraction.git `

## Task

A graph based path planner plans the path to use. A combination of Lane Detection with OpenCV and obstacle avoidance is used to get to the locations set by the planner.

To run the solution, run the following command (goal_x and goal_y are the coordinate goal, assumed to be 1.38 and 2.08 respectively if not provided):

` roslaunch parc_phase2_solution task_solution.launch goal_x:=<goal_x> goal_y:=<goal_y> `

## Challenges Faced

* Lane following code had to be rewritten due to lighting issues with physical robot.

* Little background in ROS and computer vision meant the team had a lot of ground to cover.
