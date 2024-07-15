import time
import pygame
import numpy as np
import sys
import cvxpy as cp
import pandas as pd
import motion_planner as mp

# 常量定义
K = 100.0
d_obs = 40.0
v_max = 300.0  # 速度限制


# 计算障碍物距离函数 h(x)
def h(x, obstacles, d_obs):
    h = []
    for obs in obstacles:
        h.append(np.linalg.norm(x - obs['position']) - d_obs)
    return h


# 计算 h(x) 的梯度
def grad_h(x, obstacles):
    grad_h = []
    for obs in obstacles:
        grad_h.append((x - obs['position']) / np.linalg.norm(x - obs['position']))
    return grad_h


# 期望速度函数
def v_des(x, x_goal):
    return -K * (x - x_goal)


# 用于避障的 QP 解算器
def qp_solver(x, x_goal, obstacles, alpha, d_obs, v_max):
    v = cp.Variable(2)
    v_desired = v_des(x, x_goal)

    h_values = h(x, obstacles, d_obs)
    grad_h_values = grad_h(x, obstacles)

    constraints = []
    for h_val, grad_h_val in zip(h_values, grad_h_values):
        constraints.append(grad_h_val @ v >= -alpha * h_val)

    # 添加速度限制
    constraints.append(cp.norm(v, 2) <= v_max)

    objective = cp.Minimize(cp.sum_squares(v - v_desired))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    return v.value


# 生成随机障碍物
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


pygame.init()

screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("particle simulation control")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos_a_star1 = np.array([50.0, 50.0])
particle_pos_a_star2 = np.array([50.0, 50.0])
user_goal = np.array([50.0, 50.0])
user_goal_a_star2 = np.array([50.0, 50.0])
num_obstacles = 10
particle_speed = 200
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos)
boundary_thickness = 2
boundaries = [
    {'position': np.array([screen_width / 2, boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width / 2, screen_height - boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width - boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness}
]

delta_t = 0.01
running = True

planner = mp.MotionPlanner(grid_size=500, grid_step=5)
path_a_star1 = planner.a_star(start_pos, goal_pos, obstacles)
path_a_star2 = planner.a_star(start_pos, goal_pos, obstacles)
path_index_a_star1 = 0
path_index_a_star2 = 0
trajectory_a_star1 = []
trajectory_a_star2 = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE)

    # 更新粒子1的位置
    if path_index_a_star1 < len(path_a_star1):
        a_star_goal1 = np.array(path_a_star1[path_index_a_star1])
        direction_a_star1 = a_star_goal1 - particle_pos_a_star1
        distance_a_star1 = np.linalg.norm(direction_a_star1)
        if distance_a_star1 < 5:
            path_index_a_star1 += 1
        else:
            direction_a_star1 = direction_a_star1 / distance_a_star1 * particle_speed * delta_t
            particle_pos_a_star1 += direction_a_star1
        user_goal_a_star1 = a_star_goal1
    else:
        user_goal_a_star1 = goal_pos

    # 更新粒子2的位置
    if path_index_a_star2 < len(path_a_star2):
        user_goal = np.array(path_a_star2[path_index_a_star2])
        v = qp_solver(particle_pos_a_star2, user_goal, obstacles, alpha=2.0, d_obs=d_obs, v_max=v_max)
        particle_pos_a_star2 += v * delta_t
        path_index_a_star2 += 1
    else:
        user_goal_a_star2 = goal_pos
        v = qp_solver(particle_pos_a_star2, goal_pos, obstacles, alpha=2.0, d_obs=d_obs, v_max=v_max)
        particle_pos_a_star2 += v * delta_t
    particle_pos_a_star1[0] = np.clip(particle_pos_a_star1[0], 0, screen_width)
    particle_pos_a_star1[1] = np.clip(particle_pos_a_star1[1], 0, screen_height)
    particle_pos_a_star2[0] = np.clip(particle_pos_a_star2[0], 0, screen_width)
    particle_pos_a_star2[1] = np.clip(particle_pos_a_star2[1], 0, screen_height)

    trajectory_a_star1.append(particle_pos_a_star1.copy())
    trajectory_a_star2.append(particle_pos_a_star2.copy())
    print(trajectory_a_star2)
    if np.linalg.norm(particle_pos_a_star1 - goal_pos) < 5 and np.linalg.norm(particle_pos_a_star2 - goal_pos) < 5:
        print("Particle reached goal")
        running = False

    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)
    pygame.draw.circle(screen, BLUE, particle_pos_a_star1.astype(int), 5)
    pygame.draw.circle(screen, RED, particle_pos_a_star2.astype(int), 5)

    for j in range(1, len(trajectory_a_star1)):
        pygame.draw.line(screen, BLUE, trajectory_a_star1[j - 1], trajectory_a_star1[j], 2)
    for j in range(1, len(trajectory_a_star2)):
        pygame.draw.line(screen, RED, trajectory_a_star2[j - 1], trajectory_a_star2[j], 2)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
