from collections import PriorityQueue

class OccupancyGridGraph(object):
    def __init__(self, occ_grid):
        self.occ_grid = occ_grid
        self.rows = len(occ_grid)
        self.cols = len(occ_grid[0])

    def neighbors(self, node, threshold):
        row, col = node
        node_neighbors = []
        if row != 0:
            # up
            if self.occ_grid[row - 1, col] < threshold:
                node_neighbors.append((row - 1, col))
            if col != 0:
                # left
                if self.occ_grid[row, col - 1] < threshold:
                    node_neighbors.append((row, col - 1))
                # up left
                if self.occ_grid[row - 1, col - 1] < threshold:
                    node_neighbors.append((row - 1, col - 1))
            if col != self.cols - 1:
                # right
                if self.occ_grid[row, col + 1] < threshold:
                    node_neighbors.append((row, col + 1))
                # up right
                if self.occ_grid[row - 1, col + 1] < threshold:
                    node_neighbors.append((row - 1, col + 1))
         if row != self.rows - 1:
            # down
            if self.occ_grid[row + 1, col] < threshold:
                node_neighbors.append((row + 1, col))
            if col != 0:
                # down left
                if self.occ_grid[row + 1, col - 1] < threshold:
                    node_neighbors.append((row + 1, col - 1))
            if col != self.cols - 1:
                # down right
                if self.occ_grid[row + 1, col + 1] < threshold:
                    node_neighbors.append((row + 1, col + 1))
        return node_neighbors

    def heuristic(a, b):
        (x1, y1) = a
        (x2, y2) = b
        return abs(x1 - x2) + abs(y1 - y2)

    def a_star_search(graph, start, goal, threshold):
        frontier = PriorityQueue()
        frontier.put(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        while not frontier.empty():
            current = frontier.get()
            
            if current == goal:
                break
            
            for next in graph.neighbors(current, threshold):
                new_cost = cost_so_far[current] + 1 #graph.cost(current, next)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + heuristic(goal, next)
                    frontier.put(next, priority)
                    came_from[next] = current
        
        return came_from, cost_so_far

    def came_from_to_path(self, came_from, goal):
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        return reversed(path)