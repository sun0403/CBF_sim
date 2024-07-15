
import pygame
import numpy as np
import time
import sys
import motion_planner as mp

np.random.seed(3)
K_attr = 80.0
K_rep = 80.0
rho0 = 10.0

def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']

def grad_U_pot(x, x_goal, obs, rho_0):
    grad_U_attr = K_attr * (x - x_goal)
    grad_U_obs = np.zeros_like(x, dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            rho_x = np.maximum(1e-2, rho_x)
            grad_U_obs += K_rep * (1 / rho_x - 1 / rho_0) * (-1 / rho_x ** 2) * (x - ob['position']) / (np.linalg.norm(x - ob['position']) + 1e-6)
    return -(grad_U_attr + grad_U_obs)

def generate_random_obstacles(num_obstacles, start_pos, goal_pos, field_size=500):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            position = np.random.rand(2) * field_size
            new_obstacle = {'position': position, 'radius': 50}
            if (np.linalg.norm(position - start_pos) > new_obstacle['radius'] and
                np.linalg.norm(position - goal_pos) > new_obstacle['radius'] and
                all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in obstacles)):
                obstacles.append(new_obstacle)
                break
    return obstacles

np.random.seed(1)
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
goal_pos = np.array([450.0, 350.0])
particle_pos_a_star1 = np.array([50.0, 50.0])
particle_pos_a_star2 = np.array([50.0, 50.0])
user_goal = np.array([50.0, 50.0])
num_obstacles = 20
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

planner = mp.MotionPlanner(grid_size=500, grid_step=10)
path_a_star1 = planner.a_star(start_pos, goal_pos, obstacles)
path_a_star2 = planner.a_star(start_pos, goal_pos, obstacles)
path_index_a_star1 = 1
path_index_a_star2 = 1
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

            user_goal2 = path_a_star2[path_index_a_star2]
            F = grad_U_pot(particle_pos_a_star2, user_goal2, obstacles, rho0)
            particle_pos_a_star2 += F * delta_t
            path_index_a_star2 += 1
    else:
        user_goal_a_star2 = goal_pos

    particle_pos_a_star1[0] = np.clip(particle_pos_a_star1[0], 0, screen_width)
    particle_pos_a_star1[1] = np.clip(particle_pos_a_star1[1], 0, screen_height)
    particle_pos_a_star2[0] = np.clip(particle_pos_a_star2[0], 0, screen_width)
    particle_pos_a_star2[1] = np.clip(particle_pos_a_star2[1], 0, screen_height)

    trajectory_a_star1.append(particle_pos_a_star1.copy())
    trajectory_a_star2.append(particle_pos_a_star2.copy())

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
