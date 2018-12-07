#!/usr/bin/env python
import rospy
import tf2_ros

from geometry_msgs.msg import Point
from ar_track_alvar_msgs.msg import AlvarMarkers

class TurtlebotPose:
  def __init__(self):
    #Initialize the node
    rospy.init_node('turtlebot_pose_publisher')

    self.tfBuffer = tf2_ros.Buffer()
    self.tfListener = tf2_ros.TransformListener(self.tfBuffer)

    self.pose_pub = rospy.Publisher("/turtlebot_pose1", Point, queue_size=1)

    self.turtlebot_frame = 'base_link'
    self.origin_frame = 'ar_marker_14'


  def run(self):
    while not rospy.is_shutdown():
      try:
        trans = self.tfBuffer.lookup_transform(self.origin_frame, self.turtlebot_frame, rospy.Time())
        x, y = trans.transform.translation.x, trans.transform.translation.y
        point = Point(y, x, 0) # flip because the origin frame has inverted x and y
        self.pose_pub.publish(point)
      except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
          print('EXCEPTION:', e)
          pass
if __name__ == '__main__':
  node = TurtlebotPose()
  node.run()