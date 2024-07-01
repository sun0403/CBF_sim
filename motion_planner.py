import heapq

import numpy as np


def a_star(start, goal, obstacles, grid_size=500, grid_step=1):
    def heuristic(a, b):
        return np.linalg.norm(a - b)

    def is_free(pos):
        for obs in obstacles:
            if np.linalg.norm(pos - obs['position']) <= obs['radius']:
                return False
        return True

    start = tuple(start.astype(int))
    goal = tuple(goal.astype(int))

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(np.array(start), np.array(goal))}

    while open_set:
        _, current = heapq.heappop(open_set)
        if np.linalg.norm(np.array(current) - np.array(goal)) < grid_step:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx in range(-grid_step, grid_step + 1, grid_step):
            for dy in range(-grid_step, grid_step + 1, grid_step):
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size and is_free(np.array(neighbor)):
                    tentative_g_score = g_score[current] + heuristic(np.array(current), np.array(neighbor))
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(np.array(neighbor), np.array(goal))
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []