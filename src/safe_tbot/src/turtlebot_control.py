#!/usr/bin/env python
#The line above tells Linux that this file is a Python script,
#and that the OS should use the Python interpreter in /usr/bin/env
#to run it. Don't forget to use "chmod +x [filename]" to make
#this script executable.

#Import the rospy package. For an import to work, it must be specified
#in both the package manifest AND the Python file in which it is used.
import rospy
import tf2_ros
import sys
import numpy as np

import time
from std_msgs.msg import Empty

from geometry_msgs.msg import Twist, Vector3, Point
from visualization_msgs.msg import Marker, MarkerArray

from safe_tbot.msg import Plan
import transformations
from nav_msgs.msg import Odometry

#Define the method which contains the main functionality of the node.
class Controller:
  def __init__(self):
    #Run this program as a new node in the ROS computation graph 
    #called /turtlebot_controller.
    rospy.init_node('turtlebot_controller', anonymous=True)

    self.load_parameters()

    self.turtlebot_frame = 'base_link'
    self.origin_frame = 'ar_marker_14'
    self.tfBuffer = tf2_ros.Buffer()
    self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

    self.pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)
    self.r = rospy.Rate(10)
    self.K1 = 0.5
    self.K2 = 1.

    rospy.Subscriber("/turtlebot_plan", Plan, self.planReceived)
    # rospy.Subscriber("/turtlebot_pose1", Point, self.turtlebot_pose_received)  
    rospy.Subscriber("/odom", Odometry, self.odom_received)


    #begin odom added code#

    ## reset the odometry frame when the turtlebot starts

    # set up the odometry reset publisher
    reset_odom = rospy.Publisher('/mobile_base/commands/reset_odometry', Empty, queue_size=10)

    self.current_waypoint_pub = rospy.Publisher('/current_waypoint', MarkerArray, queue_size=10)

    # reset odometry (these messages take a few iterations to get through)
    timer = time.time()
    while time.time() - timer < 15:
        reset_odom.publish(Empty())

    # end odom added code #

    self.current_plan = None

  def load_parameters(self):
    # resolution (m/cell)
    # self.res = rospy.get_param("pred/resolution")
    self.res = 0.1464

    # closeness threshold for waypoint planning (m)
    # self.closeness = rospy.get_param("state/closeness")
    self.closeness = 0.2

    # turtlebot start and current positions (m)
    self.turtlebot_start = rospy.get_param("state/turtlebot_start")
    self.turtlebot_x, self.turtlebot_y = self.turtlebot_start
    self.turtlebot_theta = 0

    # max speeds
    self.max_linear_speed = 0.3
    self.max_angular_speed = 0.3
     

  def turtlebot_pose_received(self, point_msg):
    self.turtlebot_x, self.turtlebot_y = point_msg.x, point_msg.y
  
  def odom_received(self, odom_msg):
    position = odom_msg.pose.pose.position
    self.turtlebot_x, self.turtlebot_y = position.x, position.y

    quat = odom_msg.pose.pose.orientation          
    quat = np.array([quat.x, quat.y, quat.z, quat.w])
    euler = transformations.euler_from_quaternion(quat)
    turtlebot_odom_angle = euler[1]
    self.turtlebot_theta = turtlebot_odom_angle

  def planReceived(self, msg):
    print('CONTROLLER RECEIVED PLAN')
    self.current_plan = msg.plan
    # print('NEW PLAN:')
    # for pt in msg.plan:
    #   print(pt.x, pt.y)
    self.start_timestep = msg.stamp
    self.current_waypoint = 1 #0

  def state_to_marker(self, xy=[0,0], color=[1.0,0.0,0.0]):
    """
    Converts xy position to marker type to vizualize human
    """
    marker = Marker()
    marker.header.frame_id = "/world"
    marker.header.stamp = rospy.Time().now()

    marker.type = marker.SPHERE
    marker.action = marker.ADD
    marker.pose.orientation.w = 1
    marker.pose.position.z = 0
    marker.scale.x = self.res
    marker.scale.y = self.res
    marker.scale.z = self.res
    marker.color.a = 1.0
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]

    marker.pose.position.x = xy[0]
    marker.pose.position.y = xy[1]

    return marker

  def publish_points_as_markers(self, points, publisher, color):
    marker_array = MarkerArray()
    for pt in points:
      if hasattr(pt, 'x') and hasattr(pt, 'y'):
        xy = [pt.x, pt.y]
      else:
        xy = [pt[0], pt[1]]
      marker = self.state_to_marker(xy=xy, color=color)
      marker_array.markers.append(marker)

    # Re-number the marker IDs
    id = 0
    for m in marker_array.markers:
      m.id = id
      id += 1

    publisher.publish(marker_array)

  def run(self):
    while not rospy.is_shutdown():
      if self.current_plan is not None:
        try:
          goal_x, goal_y = self.current_plan[self.current_waypoint].x, self.current_plan[self.current_waypoint].y # real (m)
          # get goal in real coordinates (m)
          # goal_x = goal_row * self.res + 0.5 * self.res
          # goal_y = goal_col * self.res + 0.5 * self.res

          # print('current waypoint real:', goal_x, goal_y)
          goal_point = Point(goal_x, goal_y, 0.2)
          self.publish_points_as_markers([goal_point], self.current_waypoint_pub, [1., 1., 0.])
          
          # trans = self.tfBuffer.lookup_transform(self.origin_frame, self.turtlebot_frame, rospy.Time())
          # # turtlebot_curr_y, turtlebot_curr_x = trans.transform.translation.x, trans.transform.translation.y  # flip because the origin frame has inverted x and y
          # quat = trans.transform.rotation          
          # quat = np.array([quat.x, quat.y, quat.z, quat.w])
          # euler = transformations.euler_from_quaternion(quat)
          # turtlebot_ar_angle = euler[1]
          # print('turtlebot theta:', turtlebot_ar_angle)
          
          # Process trans to get your state error
          goal_angle = np.arctan2(goal_y - self.turtlebot_y, goal_x - self.turtlebot_x) # get the angle to the goal (assumes self.origin_frame is at (0, 0))
          # angle_diff = goal_angle - turtlebot_ar_angle
          angle_diff = goal_angle - self.turtlebot_theta
          # distance between turtlebot and goal
          dist = np.sqrt((goal_x - self.turtlebot_x)**2 + (goal_y - self.turtlebot_y)**2)
          # get the state errors in the turtlebot frame
          total_x = dist * np.sin(abs(angle_diff))
          total_y = dist * np.cos(abs(angle_diff))
          # total_x = self.turtlebot_x + goal_x 
          # total_y = self.turtlebot_y + goal_y
          # print('turtlebot x err, y err:', total_x, total_y)
          
          if abs(total_x) < self.closeness and abs(total_y) < self.closeness:
            self.current_waypoint = min(len(self.current_plan) - 1, self.current_waypoint + 1)
            print('self.current_waypoint:', self.current_waypoint)


          # make sure the AR tag stays visible by enforcing that the z axes of the camera and the ar tag are somewhat aligned

          xdot = self.K1 * (total_x)
          if abs(xdot) > self.max_linear_speed:
            xdot = np.sign(xdot) * self.max_linear_speed
          thetadot = self.K2 * (total_y)
          # print('PROPOSED_THETADOT', total_y)
          if abs(thetadot) > self.max_angular_speed:
            thetadot = np.sign(thetadot) * self.max_angular_speed
          # if abs(self.turtlebot_theta) > (np.pi / 4):
          #   thetadot = 0
          # Generate a control command to send to the robot
          
          # thetadot = 
          print("GOAL ANGLE", goal_angle)
          print('PROPOSED_THETADOT', thetadot)



          linear = Vector3(xdot, 0, 0)
          angular = Vector3(0, 0, thetadot)
          control_command = Twist(linear, angular)
          # print('control:', control_command)
          # #################################### end your code ###############

          self.pub.publish(control_command)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
          print('EXCEPTION:', e)
          pass
      else:
        print('PLAN IS NONE!')

      
# This is Python's sytax for a main() method, which is run by default
# when exectued in the shell
if __name__ == '__main__':

  try:
    control = Controller()
    control.run()
  except rospy.ROSInterruptException:
    pass
