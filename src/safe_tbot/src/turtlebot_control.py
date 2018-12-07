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

from geometry_msgs.msg import Twist, Vector3, Point

from safe_tbot.msg import Plan
import transformations

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
    self.K1 = 0.3
    self.K2 = 1

    rospy.Subscriber("/turtlebot_plan", Plan, self.planReceived)
    rospy.Subscriber("/turtlebot_pose1", Point, self.turtlebot_pose_received)  


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

    # max speeds
    self.max_linear_speed = 0.1
    self.max_angular_speed = 0.1
     

  def turtlebot_pose_received(self, point_msg):
    self.turtlebot_x, self.turtlebot_y = point_msg.x, point_msg.y
  
  def planReceived(self, msg):
    print('CONTROLLER RECEIVED PLAN')
    self.current_plan = msg.plan
    self.start_timestep = msg.stamp
    self.current_waypoint = 0

  def run(self):
    while not rospy.is_shutdown():
      if self.current_plan is not None:
        try:
          goal_row, goal_col = self.current_plan[self.current_waypoint].x, self.current_plan[self.current_waypoint].y
          # get goal in real coordinates (m)
          goal_x = goal_row * self.res + 0.5 * self.res
          goal_y = goal_col * self.res + 0.5 * self.res
          
          trans = self.tfBuffer.lookup_transform(self.origin_frame, self.turtlebot_frame, rospy.Time())
          # turtlebot_curr_y, turtlebot_curr_x = trans.transform.translation.x, trans.transform.translation.y  # flip because the origin frame has inverted x and y
          quat = trans.transform.rotation          
          quat = np.array([quat.x, quat.y, quat.z, quat.w])
          euler = transformations.euler_from_quaternion(quat)
          turtlebot_ar_angle = euler[1]
          print('turtlebot theta:', turtlebot_ar_angle)
          
          # Process trans to get your state error
          goal_angle = np.arctan2(goal_y, goal_x) # get the angle to the goal (assumes self.origin_frame is at (0, 0))
          angle_diff = goal_angle - turtlebot_ar_angle
          # distance between turtlebot and goal
          dist = np.sqrt((goal_x - self.turtlebot_x)**2 + (goal_y - self.turtlebot_y)**2)
          # get the state errors in the turtlebot frame
          total_x = dist * np.sin(abs(angle_diff))
          total_y = dist * np.cos(abs(angle_diff))
          # total_x = self.turtlebot_x + goal_x 
          # total_y = self.turtlebot_y + goal_y
          print('turtlebot x err, y err:', total_x, total_y)
          
          if total_x < self.closeness and total_y < self.closeness:
            self.current_waypoint = min(len(self.current_plan) - 1, self.current_waypoint + 1)

          # make sure the AR tag stays visible by enforcing that the z axes of the camera and the ar tag are somewhat aligned

          xdot = self.K1 * (total_x)
          if abs(xdot) > self.max_linear_speed:
            xdot = np.sign(xdot) * self.max_linear_speed
          thetadot = self.K2 * (total_y)
          if abs(thetadot) > self.max_angular_speed:
            thetadot = np.sign(thetadot) * self.max_angular_speed
          # Generate a control command to send to the robot
          linear = Vector3(xdot, 0, 0)
          angular = Vector3(0, 0, thetadot)
          control_command = Twist(linear, angular)
          print('control:', control_command)
          # #################################### end your code ###############

          self.pub.publish(control_command)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
          print('EXCEPTION:', e)
          pass

      
# This is Python's sytax for a main() method, which is run by default
# when exectued in the shell
if __name__ == '__main__':

  try:
    control = Controller()
    control.run()
  except rospy.ROSInterruptException:
    pass
