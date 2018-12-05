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

from geometry_msgs.msg import Twist, Vector3

from safe_tbot.msg import Plan


#Define the method which contains the main functionality of the node.
class Controller:
  def __init__(self, turtlebot_frame, origin_frame):
    self.turtlebot_frame = turtlebot_frame
    self.origin_frame = origin_frame
    self.tfBuffer = tf2_ros.Buffer()
    self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

    self.pub = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=10)
    self.r = rospy.Rate(10)
    self.K1 = 0.3
    self.K2 = 1

    rospy.Subscriber("/turtlebot_plan", Plan, self.planReceived)  

    self.current_plan = None

  def load_parameters(self):
    # resolution (m/cell)
    self.res = rospy.get_param("pred/resolution")

    # closeness threshold for waypoint planning (m)
    self.closeness = rospy.get_param("state/closeness")

  def planReceived(self, msg):
    self.current_plan = msg.plan
    self.start_timestep = msg.stamp
    self.current_waypoint = 0

  def run(self):
    while not rospy.is_shutdown():
      try:
        goal_row, goal_col = self.current_plan[self.current_waypoint].x, self.current_plan[self.current_waypoint].y
        x_disp = goal_row * self.res + 0.5 * self.res
        y_disp = goal_row * self.res + 0.5 * self.res
        
        trans = self.tfBuffer.lookup_transform(self.turtlebot_frame, self.origin_frame, rospy.Time())
        print('trans:', trans)
        # Process trans to get your state error
        total_x = trans.transform.translation.x + x_disp
        total_y = trans.transform.translation.y + y_disp
        
        if total_x < self.closeness and total_y < self.closeness:
          self.current_waypoint = min(len(self.current_plan), self.current_waypoint + 1)

        xdot = self.K1 * (total_x)
        thetadot = self.K2 * (total_y)
        # Generate a control command to send to the robot
        linear = Vector3(xdot, 0, 0)
        angular = Vector3(0, 0, thetadot)
        control_command = Twist(linear, angular)
        print('control:', control_command)
        #################################### end your code ###############

        self.pub.publish(control_command)
      except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        pass
      # Use our rate object to sleep until it is time to publish again
      r.sleep()

      



  
  


      
# This is Python's sytax for a main() method, which is run by default
# when exectued in the shell
if __name__ == '__main__':
  # Check if the node has received a signal to shut down
  # If not, run the talker method

  #Run this program as a new node in the ROS computation graph 
  #called /turtlebot_controller.
  rospy.init_node('turtlebot_controller', anonymous=True)

  try:
    control = Controller(sys.argv[1], sys.argv[2])
    control.run()
  except rospy.ROSInterruptException:
    pass
