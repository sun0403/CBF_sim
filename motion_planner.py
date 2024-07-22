import numpy as np
import heapq
from collections import deque
import random

class MotionPlanner:
    def __init__(self, grid_size=500, grid_step=1):
        self.grid_size = grid_size
        self.grid_step = grid_step

    def is_free(self, pos, obstacles):
        for obs in obstacles:
            if np.linalg.norm(pos - obs['position']) <= obs['radius']:
                return False
        return True

    def select_random_planner(self, start_pos, goal_pos, obstacles):
        methods = ['a_star', 'bfs','rrt']
        selected_method = random.choice(methods)
        if selected_method == 'a_star':
            return self.a_star(start_pos, goal_pos, obstacles)
        elif selected_method == 'rrt':
            return self.rrt(start_pos, goal_pos, obstacles)
        elif selected_method == 'bfs':
            return self.bfs(start_pos, goal_pos, obstacles)
    def a_star(self, start, goal, obstacles):
        def heuristic(a, b):
            return np.linalg.norm(a - b)

        start = tuple(start.astype(int))
        goal = tuple(goal.astype(int))

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(np.array(start), np.array(goal))}

        while open_set:
            _, current = heapq.heappop(open_set)
            if np.linalg.norm(np.array(current) - np.array(goal)) < self.grid_step:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dx in range(-self.grid_step, self.grid_step + 1, self.grid_step):
                for dy in range(-self.grid_step, self.grid_step + 1, self.grid_step):
                    neighbor = (current[0] + dx, current[1] + dy)
                    if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size and self.is_free(np.array(neighbor), obstacles):
                        tentative_g_score = g_score[current] + heuristic(np.array(current), np.array(neighbor))
                        if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                            came_from[neighbor] = current
                            g_score[neighbor] = tentative_g_score
                            f_score[neighbor] = tentative_g_score + heuristic(np.array(neighbor), np.array(goal))
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []

    def bfs(self, start, goal, obstacles):
        start = tuple(start.astype(int))
        goal = tuple(goal.astype(int))

        queue = deque([start])
        came_from = {}
        came_from[start] = None

        while queue:
            current = queue.popleft()
            if np.linalg.norm(np.array(current) - np.array(goal)) < self.grid_step:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            for dx in range(-self.grid_step, self.grid_step + 1, self.grid_step):
                for dy in range(-self.grid_step, self.grid_step + 1, self.grid_step):
                    neighbor = (current[0] + dx, current[1] + dy)
                    if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size and self.is_free(np.array(neighbor), obstacles) and neighbor not in came_from:
                        queue.append(neighbor)
                        came_from[neighbor] = current

        return []

    def rrt(self, start, goal, obstacles, max_samples=1024, goal_sample_rate=0.1):
        def get_random_point():
            if np.random.rand() < goal_sample_rate:
                return goal
            return np.array([np.random.uniform(0, self.grid_size), np.random.uniform(0, self.grid_size)])

        def get_nearest_node(nodes, point):
            distances = [np.linalg.norm(node - point) for node in nodes]
            nearest_index = np.argmin(distances)
            return nodes[nearest_index], nearest_index

        def is_collision_free(point1, point2, obstacles):
            direction = point2 - point1
            distance = np.linalg.norm(direction)
            direction = direction / distance

            for i in range(int(distance // self.grid_step)):
                intermediate_point = point1 + direction * i * self.grid_step
                if not self.is_free(intermediate_point, obstacles):
                    return False
            return self.is_free(point2, obstacles)

        import time
        t1 = time.time()

        start = np.array(start)
        goal = np.array(goal)
        nodes = [start]
        parents = {0: None}

        for _ in range(max_samples):
            rand_point = get_random_point()
            nearest_node, nearest_index = get_nearest_node(nodes, rand_point)
            direction = (rand_point - nearest_node) / np.linalg.norm(rand_point - nearest_node)
            new_node = nearest_node + direction * self.grid_step
            new_node = np.clip(new_node, 0, self.grid_size)

            if is_collision_free(nearest_node, new_node, obstacles):
                nodes.append(new_node)
                parents[len(nodes) - 1] = nearest_index

                if np.linalg.norm(new_node - goal) < self.grid_step:
                    nodes.append(goal)
                    parents[len(nodes) - 1] = len(nodes) - 2
                    break

        print(f"Time needed: {time.time() - t1}")

        path = []
        current_index = len(nodes) - 1
        while current_index is not None:
            path.append(nodes[current_index])
            current_index = parents[current_index]
        path.reverse()

        return path