from Queue import PriorityQueue
import pdb

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import OrderedDict

class OccupancyGridGraph(object):
    def __init__(self, occ_grids, resolution, robot_radius, num_collision_samples):
        """
        occ_grids: (timestep, row, column)
        resolution: resolution
        """
        self.occ_grids = occ_grids
        self.max_time, self.rows, self.cols = occ_grids.shape
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.num_collision_samples = num_collision_samples

    def neighbors(self, node, threshold):
        time, row, col = node
        assert time >= 0 and time < self.max_time
        assert row >= 0 and row < self.rows
        assert col >= 0 and col < self.cols
        node_neighbors = []
        if time < self.max_time - 1: # last timestamp doesn't have neighbors
            future_occ_grid = self.occ_grids[time + 1]
            # stay in place, go forward in time
            if self.simple_check_no_collision(
                    (time + 1, row, col), 
                    self.robot_radius, 
                    self.num_collision_samples, future_occ_grid, 
                    threshold):
                node_neighbors.append(((time + 1, row, col), 0)) # no cost
            if col != 0:
                # left
                if self.simple_check_no_collision((time + 1, row, col - 1), self.robot_radius,
                        self.num_collision_samples, future_occ_grid, threshold):
                    node_neighbors.append(((time + 1, row, col - 1), 1)) # cost = 1
            if col != self.cols - 1:
                # right
                if self.simple_check_no_collision((time + 1, row, col + 1), self.robot_radius,
                        self.num_collision_samples, future_occ_grid, threshold):
                    node_neighbors.append(((time + 1, row, col + 1), 1)) # cost = 1
            if row != 0:
                # up
                if self.simple_check_no_collision((time + 1, row - 1, col), self.robot_radius,
                        self.num_collision_samples, future_occ_grid, threshold):
                    node_neighbors.append(((time + 1, row - 1, col), 1)) # cost = 1
                if col != 0:
                    # up left
                    if self.simple_check_no_collision((time + 1, row - 1, col - 1), self.robot_radius,
                            self.num_collision_samples, future_occ_grid, threshold):
                        node_neighbors.append(((time + 1, row - 1, col - 1), np.sqrt(2))) # cost = sqrt(2)
                if col != self.cols - 1:
                    # up right
                    if self.simple_check_no_collision((time + 1, row - 1, col + 1), self.robot_radius,
                            self.num_collision_samples, future_occ_grid, threshold):
                        node_neighbors.append(((time + 1, row - 1, col + 1), np.sqrt(2))) # cost = sqrt(2)
            if row != self.rows - 1:
                # down
                if self.simple_check_no_collision((time + 1, row + 1, col), self.robot_radius,
                        self.num_collision_samples, future_occ_grid, threshold):
                    node_neighbors.append(((time + 1, row + 1, col), 1)) # cost = 1
                if col != 0:
                    # down left
                    if self.simple_check_no_collision((time + 1, row + 1, col - 1), self.robot_radius,
                            self.num_collision_samples, future_occ_grid, threshold):
                        node_neighbors.append(((time + 1, row + 1, col - 1), np.sqrt(2))) # cost = sqrt(2)
                if col != self.cols - 1:
                    # down right
                    if self.simple_check_no_collision((time + 1, row + 1, col + 1), self.robot_radius,
                            self.num_collision_samples, future_occ_grid, threshold):
                        node_neighbors.append(((time + 1, row + 1, col + 1), np.sqrt(2))) # cost = sqrt(2)
        return node_neighbors

    def check_no_collision(self, grid_cell, radius, num_samples, occ_grid, threshold):
        """Return True if there is no collision probabilistically using the given
        threshold.

        Sample n points in the circle, set each of their values to the value
        of the occupancy grid cell they fall in, and divide the sum by n. This
        will give an approximate integral of the circle over the occupancy grid.

        grid_cell: (t, x, y) of the grid cell
        """
        t, grid_x, grid_y = grid_cell
        centroid = self.grid_cell_centroid(grid_x, grid_y)
        circle_x, circle_y = self.sample_from_circle(num_samples, centroid, radius)
        grid_circle_x, grid_circle_y = self.point_to_grid_cell(circle_x, circle_y)
        integral = sum([occ_grid[x, y] for x, y in zip(grid_circle_x, grid_circle_y)]) / float(num_samples)
        return integral < threshold

    def simple_check_no_collision(self, grid_cell, radius, num_samples, occ_grid, threshold):
        t, row, col = grid_cell
        return occ_grid[row, col] < threshold

    def grid_cell_centroid(self, grid_x, grid_y):
        """Return the centroid of the given grid cell.
        Assume the origin is at (0, 0).
        """
        point_x = grid_x * self.resolution + (self.resolution / 2.)
        point_y = grid_y * self.resolution + (self.resolution / 2.)
        return point_x, point_y
    
    def point_to_grid_cell(self, x, y):
        """Return which grid cell the given point is in."""
        grid_x = x / self.resolution
        grid_y = y / self.resolution
        if type(grid_x) == np.ndarray:
            grid_x = grid_x.astype(int)
        else:
            grid_x = int(grid_x)
        if type(grid_y) == np.ndarray:
            grid_y = grid_y.astype(int)
        else:
            grid_y = int(grid_y)
        grid_x = np.maximum(np.minimum(grid_x, self.rows - 1), 0)
        grid_y = np.maximum(np.minimum(grid_y, self.cols - 1), 0)
        return grid_x, grid_y

    def sample_from_circle(self, num_samples, center, radius):
        rho = np.sqrt(np.random.random(num_samples)) * radius # sample sqrt of radius
        theta = np.random.random(num_samples) * 2 * np.pi
        centerx, centery = center
        x = rho * np.cos(theta) + centerx
        y = rho * np.sin(theta) + centery
        return x, y

def heuristic(a, b):
    """Manhattan distance heuristic."""
    (t1, x1, y1) = a
    (t2, x2, y2) = b
    return abs(t1 - t2) + abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph, start, goal, threshold):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = OrderedDict()
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next, cost in graph.neighbors(current, threshold):
            new_cost = cost_so_far[current] + cost #graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def came_from_to_path(came_from, goal):
    path = []
    curr = goal
    print('goalll in came from to path', goal)
    while curr is not None:
        path.append(curr)
        curr = came_from[curr]
    return list(reversed(path))

def plot_path(path, occ_grids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ts = [p[0] for p in path]
    xs = [p[1] for p in path]
    ys = [p[2] for p in path]
    ax.plot(xs, ys, zs=ts)
    ax.scatter(xs, ys, zs=ts)
    tdim, xdim, ydim = occ_grids.shape
    plt.xlim(0, xdim - 1)
    plt.ylim(0, ydim - 1)
    # plt.zlim(0, tdim - 1)
    plt.show()

def test1():
    t = 20
    rows = 9
    cols = 10

    resolution = 1.
    robot_radius = 2.5
    num_collision_samples = 1000
    
    start = (0, 0, 0)
    
    goal = (t - 1, rows - 1, cols - 1)
    # goal = (t - 1, 0, 0)
    
    threshold = 0.5
    
    for i in range(10):
        occ_grids = np.random.random((t, rows, cols))
        # occ_grids = np.zeros((t, rows, cols))
        # occ_grids = np.ones((t, rows, cols))
        # occ_grids[:, 0, 0] = 0
        # occ_grids[:2, :2, :2] = 1.


        graph = OccupancyGridGraph(occ_grids, resolution, robot_radius, num_collision_samples)
    
        came_from, cost_so_far = a_star_search(graph, start, goal, threshold)
        if goal not in came_from:
            print('No path available.')
        else:
            path = came_from_to_path(came_from, goal)
            print('path:', path)
            plot_path(path, occ_grids)
            break

if __name__ == '__main__':
    test1()