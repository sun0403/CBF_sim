import pygame
import numpy as np
import time
import sys
import motion_planner as mp

np.random.seed(12)
K_attr = 10.0
K_rep = 10.0
rho0 = 50.0

def rho(x, obs):
    """Calculate the distance from the particle to the edge of the obstacle."""
    return np.linalg.norm(x - obs['position']) - obs['radius']

def grad_U_pot(x, x_goal, obs, rho_0):
    """Calculate the gradient of the potential field."""
    grad_U_attr = K_attr * (x - x_goal)
    grad_U_obs = np.zeros_like(x, dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            rho_x = np.maximum(1e-2, rho_x)
            grad_U_obs += K_rep * (1 / rho_x - 1 / rho_0) * (-1 / rho_x ** 2) * (x - ob['position']) / (np.linalg.norm(x - ob['position']) + 1e-6)
    return -(grad_U_attr + grad_U_obs)

def generate_random_obstacles(num_obstacles, start_pos, goal_pos, d_obs, field_size):
    """Generate a list of random obstacles ensuring they don't overlap with the start and goal positions."""
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            position = np.random.rand(2) * field_size
            new_obstacle = {'position': position, 'radius': 50}
            if (np.linalg.norm(position - start_pos) > (new_obstacle['radius'] + d_obs) and
                    np.linalg.norm(position - goal_pos) > (new_obstacle['radius'] + d_obs) and
                    all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in obstacles)):
                obstacles.append(new_obstacle)
                break
    return obstacles

def angle_between(v1, v2):
    """Calculate the angle between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

# Initialize Pygame
pygame.init()

screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Simulation Control")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos = np.array([50.0, 50.0])
num_obstacles = 10
particle_speed = 200
d_obs = 10
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos, d_obs, screen_height)
angle_threshold = np.pi * 2 / 3  # Set angle threshold to 120 degrees
delta_t = 0.01
v_max = 500.0
running = True

# Initialize the motion planner
planner = mp.MotionPlanner(grid_size=500, grid_step=5)
path_2 = np.array(planner.select_random_planner(start_pos, goal_pos, obstacles))

path_index = 0
trajectory = []
all_paths = [path_2[:path_index + 1].tolist()]  # Initialize to include start point
user_goal_path = []

# PID controller variables
previous_error = 0
total_error = 0

k_p = 30
k_d = 20
k_i = 0.1

# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill(WHITE)

    if path_index < len(path_2) - 1:
        user_goal = path_2[path_index + 1]
        user_goal_path.append(user_goal)

        F = grad_U_pot(particle_pos, user_goal, obstacles, rho0)

        F[0] = np.clip(F[0], -v_max, v_max)
        F[1] = np.clip(F[1], -v_max, v_max)

        print(f'F = {F}')

        F_direction = F / np.linalg.norm(F)
        if np.linalg.norm(F_direction) == 0:
            F_direction = np.zeros(2)
        path_direction = (user_goal - particle_pos) / np.linalg.norm(user_goal - particle_pos)
        angle_diff = angle_between(F_direction, path_direction)
        if angle_diff > angle_threshold:
            print("Replanning path...")
            all_paths.append(path_2[:path_index].tolist())  # Save traversed path segments
            new_path = np.array(planner.select_random_planner(particle_pos, goal_pos, obstacles))
            path_2 = new_path
            path_index = 0
            all_paths.append([])  # Start a new path segment
        else:
            path_index += 1
            particle_pos += F * delta_t
            all_paths[-1].append(path_2[path_index].tolist())  # Update the latest path segment
    else:
        user_goal = goal_pos
        user_goal_path.append(user_goal)
        F = grad_U_pot(particle_pos, user_goal, obstacles, rho0)
        particle_pos += F * delta_t

    particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
    particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

    trajectory.append(particle_pos.copy())

    if np.linalg.norm(particle_pos - goal_pos) < 5:
        print("Particle reached goal")

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    # Draw start and goal positions
    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)
    pygame.draw.circle(screen, BLUE, particle_pos.astype(int), 5)

    # Draw particle trajectory
    for j in range(1, len(trajectory)):
        pygame.draw.line(screen, RED, trajectory[j - 1], trajectory[j], 2)

    # Draw all path segments
    for path in all_paths:
        for j in range(1, len(path)):
            pygame.draw.line(screen, [150, 150, 150], np.array(path[j - 1]).astype(int), np.array(path[j]).astype(int), 2)

    # Draw user goal path
    for j in range(1, len(user_goal_path)):
        pygame.draw.line(screen, BLUE, user_goal_path[j - 1].astype(int), user_goal_path[j].astype(int), 2)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
