import time
import pygame
import numpy as np
import sys
import random
import motion_planner as mp

def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']

def U_rep(x, obstacles, rho_0):
    U_obs = 0
    for obs in obstacles:
        if rho(x, obs) < rho_0:
            U_obs += 0.5 * K_rep * (1/rho(x, obs) - 1/rho_0)**2
    return U_obs

def grad_U_rep(x, obstacles, rho_0):
    grad_U_obs = np.zeros_like(x, dtype=np.float64)
    for obs in obstacles:
        rho_x = rho(x, obs)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1/rho_x - 1/rho_0) * (-1/rho_x**2) * (x - obs['position']) / np.linalg.norm(x - obs['position'])
    return grad_U_obs

def U_attra(x, x_goal):
    return 0.5 * K_att * np.linalg.norm(x - x_goal)**2

def grad_U_att(x, x_goal):
    return K_att * (x - x_goal)

def h(x, obstacles, delta, rho_0):
    U_rep_value = U_rep(x, obstacles, rho_0)
    return 1 / (1 + U_rep_value) - delta

def grad_h(x, obstacles, rho_0):
    U_rep_value = U_rep(x, obstacles, rho_0)
    grad_U_rep_value = grad_U_rep(x, obstacles, rho_0)
    return -grad_U_rep_value / (1 + U_rep_value)**2

def v_star(x, x_goal, obstacles, alpha, delta, rho_0):
    v_att = -grad_U_att(x, x_goal)
    h_value = h(x, obstacles, delta, rho_0)
    grad_h_value = grad_h(x, obstacles, rho_0)
    norm_grad_h_value = np.dot(grad_h_value, grad_h_value)

    if norm_grad_h_value == 0:
        return v_att

    Phi = np.dot(grad_h_value, v_att) + alpha * h_value

    if Phi < 0:
        v = v_att + (-Phi / np.dot(grad_h_value, grad_h_value)) * grad_h_value
    else:
        v = v_att

    return v

def generate_random_obstacles(num_obstacles, start_pos, goal_pos, field_size=500):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            position = np.random.rand(2) * field_size
            new_obstacle = {'position': position, 'radius': 50}
            if (np.linalg.norm(position - start_pos) > (new_obstacle['radius'] + d_obs) and
                    np.linalg.norm(position - goal_pos) > (new_obstacle['radius'] + d_obs) and
                    all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in
                        obstacles)):
                obstacles.append(new_obstacle)
                break
    return obstacles

def angle_between(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

class MotionPlanner(mp.MotionPlanner):
    def select_random_planner(self, start_pos, goal_pos, obstacles):
        methods = ['a_star', 'rrt', 'bfs']
        selected_method = random.choice(methods)
        if selected_method == 'a_star':
            return self.a_star(start_pos, goal_pos, obstacles)
        elif selected_method == 'rrt':
            return self.rrt(start_pos, goal_pos, obstacles)
        elif selected_method == 'bfs':
            return self.bfs(start_pos, goal_pos, obstacles)

pygame.init()

screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("粒子模拟控制")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)  # 添加 ORANGE 颜色定义

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos_2 = np.array([50.0, 50.0])
user_goal_2 = np.array([50.0, 50.0])
num_obstacles = 10
particle_speed = 200
d_obs = 10
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos)
boundary_thickness = 2
boundaries = [
    {'position': np.array([screen_width / 2, boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width / 2, screen_height - boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width - boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness}
]

K_att = 50.0
K_rep = 50.0
delta = 0.001
rho_0 = 10
d_obs = 20
angle_threshold = np.pi/2 # 角度阈值，设为30度

delta_t = 0.01
running = True

planner = MotionPlanner(grid_size=500, grid_step=5)
path2 = planner.a_star(start_pos, goal_pos, obstacles)
path_index_2 = 0
trajectory_a_star2 = []
all_paths = []  # 保存所有经过的规划路径段

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE)
    if path_index_2 < len(path2):
        user_goal_2 = np.array(path2[path_index_2])

        v = v_star(particle_pos_2, user_goal_2, obstacles, alpha=2.0, delta=delta, rho_0=rho_0)
        if np.linalg.norm(v) != 0:
            v_direction = v / np.linalg.norm(v)
        else:
            v_direction = np.zeros(2)

        path_direction = (user_goal_2 - particle_pos_2) / np.linalg.norm(user_goal_2 - particle_pos_2)
        angle_diff = angle_between(v_direction, path_direction)

        if angle_diff > angle_threshold or np.linalg.norm(user_goal_2 - particle_pos_2) > 20:
            print("路径偏离，重新规划...")
            new_path = planner.rrt(particle_pos_2, goal_pos, obstacles)

            all_paths.append(path2[:path_index_2])  # 保存当前路径的已走部分
            path2 = new_path
            path_index_2 = 0
            user_goal_2 = np.array(path2[path_index_2])
            v = v_star(particle_pos_2, user_goal_2, obstacles, alpha=2.0, delta=delta, rho_0=rho_0)
        particle_pos_2 += v * delta_t
        path_index_2 += 1
    else:
        user_goal_2 = goal_pos
        v = v_star(particle_pos_2, user_goal_2, obstacles, alpha=2.0, delta=delta, rho_0=rho_0)
        particle_pos_2 += v * delta_t
        all_paths.append(path2)

    particle_pos_2[0] = np.clip(particle_pos_2[0], 0, screen_width)
    particle_pos_2[1] = np.clip(particle_pos_2[1], 0, screen_height)

    trajectory_a_star2.append(particle_pos_2.copy())
    if np.linalg.norm(particle_pos_2 - goal_pos) < 5:
        print("粒子到达目标")

    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, particle_pos_2.astype(int), 5)

    for j in range(1, len(trajectory_a_star2)):
        pygame.draw.line(screen, RED, trajectory_a_star2[j - 1].astype(int), trajectory_a_star2[j].astype(int), 2)

    # 绘制所有经过的规划路径段
    for path in all_paths:
        for j in range(1, len(path)):
            pygame.draw.line(screen, BLACK, np.array(path[j - 1]).astype(int), np.array(path[j]).astype(int), 2)


    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()