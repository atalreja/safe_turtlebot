#!/usr/bin/env python
#The line above tells Linux that this file is a Python script,
#and that the OS should use the Python interpreter in /usr/bin/env
#to run it. Don't forget to use "chmod +x [filename]" to make
#this script executable.

import time
from std_msgs.msg import Empty

#Import the rospy package. For an import to work, it must be specified
#in both the package manifest AND the Python file in which it is used.
import rospy
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from ar_track_alvar_msgs.msg import AlvarMarkers
from nav_msgs.msg import Odometry

from safe_tbot.msg import Plan
from crazyflie_human.msg import OccupancyGridTime, ProbabilityGrid

from graph import *

saved_plan = False

class Planner(object):
    """docstring for Planner"""
    def __init__(self):
        # create ROS node
        rospy.init_node('turtlebot_planner', anonymous=True)

        self.load_parameters()

        self.plan_pub = rospy.Publisher('/turtlebot_plan', Plan, queue_size=10) # WAS 1
        self.plan_vis_pub = rospy.Publisher('/turtlebot_plan_vis', MarkerArray, queue_size=1)
        self.goal_pub = rospy.Publisher('/turtlebot_goal', MarkerArray, queue_size=10)
        self.turtlebot_position_pub = rospy.Publisher('/turtlebot_position', MarkerArray, queue_size=1)

        # Listening to /turtlebot_pose1 for pose
        # self.turtlebot_pose_sub = rospy.Subscriber('/turtlebot_pose1', Point, self.receive_turtlebot_point)
        # Listening to /odom for pose
        self.turtlebot_pose_sub = rospy.Subscriber('/odom', Odometry, self.receive_turtlebot_odom)

        # self.turtlebot_pose_sub = rospy.Subscriber('/ar_pose_marker', AlvarMarkers, self.receive_ar_markers)
        
        self.occu_sub = rospy.Subscriber('/occupancy_grid_time1', OccupancyGridTime, self.plan)

        

    def load_parameters(self):
        # resolution (m/cell)
        self.res = rospy.get_param("pred/resolution")

        # robot radius (m)
        self.turtlebot_radius = rospy.get_param("state/turtlebot_radius")

        # Num timesteps into the future to predict
        self.fwd_tsteps = rospy.get_param("pred/fwd_tsteps")

        self.turtlebot_plan_color = rospy.get_param("plan/turtlebot_plan_color")

        self.turtlebot_start = rospy.get_param("state/turtlebot_start")
        self.turtlebot_floor_x, self.turtlebot_floor_y = self.turtlebot_start

        # robot goal: (time, x (m), y (m))
        self.turtlebot_goal = tuple([self.fwd_tsteps - 1] + rospy.get_param("state/turtlebot_goal"))
        print('ASSIGNED self.turtlebot_goal to', self.turtlebot_goal)

        self.collision_threshold = rospy.get_param("state/collision_threshold")

        self.turtlebot_ar_marker_id = rospy.get_param("state/turtlebot_ar_marker_id")

    # def receive_ar_markers(self, alvar_markers_msg):
    #     # print('RECEIVED ALVAR MARKERS MESSAGE')
    #     for marker in alvar_markers_msg.markers:
    #         print('marker.id:', marker.id)
    #         if marker.id == self.turtlebot_ar_marker_id:
    #             self.receive_turtlebot_point(marker.pose)
    #             return
    #     print('CANNOT FIND TURTLEBOT AR MARKER')

    def receive_turtlebot_odom(self, odom_msg):
        position = odom_msg.pose.pose.position
        self.turtlebot_floor_x, self.turtlebot_floor_y = position.x, position.y
        # print('RECEIVED TURTLEBOT POSE:', self.turtlebot_floor_x, self.turtlebot_floor_y, '\n')

    def receive_turtlebot_point(self, point_msg):
        # turtlebot position (m)
        self.turtlebot_floor_x, self.turtlebot_floor_y = point_msg.x, point_msg.y
        # print('RECEIVED TURTLEBOT POSE:', self.turtlebot_floor_x, self.turtlebot_floor_y, '\n')

    def check_row_col(self, came_from, goal):
        for node in came_from:
            c_0, c_1, c_2 = node
            if c_1 == goal[1] and c_2 == goal[2]:
                return True
        return False

    def manhat(self, node, goal):
        tn, rn, cn = node
        tg, rg, cg = goal
        return abs(rn - rg)**2 + abs(cn - cg)**2

    def filter_close_to_goal(self, came_from, goal):
        possibles = []
        for node in came_from:
            if node[0] == self.fwd_tsteps - 1:
                possibles.append(node)
        best_manhat = float('inf')
        best_node = None
        for x in possibles:
            if self.manhat(x, goal) < best_manhat:
                best_manhat = self.manhat(x, goal)
                best_node = x

        return best_node

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

    def plan(self, occu_grid_time_msg):
        print('PLANNING\n')
        prob_grids = occu_grid_time_msg.gridarray
        width = prob_grids[0].width
        height = prob_grids[0].height
        occ_grids = [prob_grid.data for prob_grid in prob_grids]
        occ_grids = np.array(occ_grids).reshape([-1, width, height])

        occ_graph = OccupancyGridGraph(occ_grids, self.res, self.turtlebot_radius, num_collision_samples=100)

        turtle_temp_time, turtle_temp_x, turtle_temp_y = self.turtlebot_goal
        turtle_temp_x, turtle_temp_y = occ_graph.point_to_grid_cell(turtle_temp_x, turtle_temp_y)
        grid_cell_turtlebot_goal = tuple([int(turtle_temp_time), turtle_temp_x, turtle_temp_y])

        # print('turtlebot start:', self.turtlebot_floor_x, self.turtlebot_floor_y)
        start_grid_x, start_grid_y = occ_graph.point_to_grid_cell(self.turtlebot_floor_x, self.turtlebot_floor_y)
        start = tuple([0, start_grid_x, start_grid_y])
        came_from, cost_so_far = a_star_search(occ_graph, start, grid_cell_turtlebot_goal, self.collision_threshold)
        # print('turtlebot_goal', grid_cell_turtlebot_goal)
        # print('came_from', came_from)
        # if grid_cell_turtlebot_goal not in came_from:
        #     print('No path available.')
        # else:
        closest_to_goal = self.filter_close_to_goal(came_from, grid_cell_turtlebot_goal)
        # print('original goal', grid_cell_turtlebot_goal)
        # print('closest node to goal', closest_to_goal)
        path_grid = came_from_to_path(came_from, closest_to_goal)
        xy_real = [occ_graph.grid_cell_centroid(grid_x, grid_y) for (t, grid_x, grid_y) in path_grid]
        points_real = [Point(x, y, 0) for (x, y) in xy_real]
        # print('points_real:')
        # for pt in points_real:
        #     print(pt)
        timestamp = prob_grids[0].header.stamp
        plan = Plan(points_real, timestamp)
        print('FINISHED PLANNING\n')
        # print('plan:', plan)
        self.plan_pub.publish(plan)

        self.publish_points_as_markers(points_real, self.plan_vis_pub, color=self.turtlebot_plan_color)

    def run(self):
        while not rospy.is_shutdown():
            goal_t, goal_x, goal_y = self.turtlebot_goal
            self.publish_points_as_markers([[goal_x, goal_y]], self.goal_pub, color=[0., 0., 0.])
            self.publish_points_as_markers([[self.turtlebot_floor_x, self.turtlebot_floor_y]], self.turtlebot_position_pub, color=[1., 0., 1.])
        # rospy.spin()


if __name__ == '__main__':
  try:
    planner = Planner()
    planner.run()
  except rospy.ROSInterruptException:
    pass
