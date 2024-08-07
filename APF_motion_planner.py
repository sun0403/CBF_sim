import pygame
import numpy as np
import time
import sys
import os
import pandas as pd
import motion_planner as mp

np.random.seed(67)

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
            grad_U_obs += K_rep * (1 / rho_x - 1 / rho_0) * (-1 / rho_x ** 2) * (x - ob['position']) / (
                    np.linalg.norm(x - ob['position']) + 1e-6)
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
                    all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius'])
                        for obs in obstacles)):
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

# Simulation parameters
screen_width, screen_height = 500, 500
start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
num_obstacles = 10
rho0 = 100.0
angle_threshold = 90
delta_t = 0.01
v_max = 500.0
K_attr = 20.0
K_rep = 40.0

# PID controller variables
k_p = 100.0
k_d = 1.0
k_i = 0.2

for i in range(20):
    data = {
        "timestamp": [],
        "particle_position": [],
        "user_goal": [],
        "velocity": [],
        "collision": [],
        "success": [],
    }

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Particle Simulation Control")

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    ORANGE = (255, 165, 0)

    particle_pos = start_pos.copy()
    obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos, d_obs=20, field_size=screen_height)

    # Initialize the motion planner
    planner = mp.MotionPlanner(grid_size=500, grid_step=4)
    path_2, planning_method = planner.select_random_planner(start_pos, goal_pos, obstacles, return_method=True)

    path_index = 0
    trajectory = []
    all_paths = [np.array(path_2[:path_index + 1])]  # Initialize to include start point
    user_goal_path = []

    # PID controller variables
    previous_error = 0
    total_error = 0

    running = True
    start_time = time.time()
    success = True  # Initialize success flag
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(WHITE)

        if path_index < len(path_2) - 1:
            # PID controller
            error_pos = particle_pos - path_2[path_index + 1]
            error_delta = error_pos - previous_error
            total_error += error_pos  # Accumulated error
            previous_error = error_pos
            u = k_p * error_pos + k_d * error_delta + k_i * total_error

            user_goal = particle_pos - u * delta_t
            user_goal_path.append(user_goal)

            v = grad_U_pot(particle_pos, user_goal, obstacles, rho_0=rho0)

            v_magnitude = np.linalg.norm(v)
            v_direction = v / (v_magnitude + 1e-5)
            if np.linalg.norm(v_direction) == 0:
                v_direction = np.zeros(2)
            if v_magnitude > v_max:
                v = v_direction * v_max

            path_direction = (user_goal - particle_pos) / np.linalg.norm(user_goal - particle_pos)
            if np.all(path_direction) == 0:
                path_direction = np.zeros(2)
            angle_diff = angle_between(v_direction, path_direction)
            if angle_diff > angle_threshold:
                print(f"Replanning path and angel diffence8{angle_diff}")
                new_path = np.array(planner.select_random_planner(particle_pos, goal_pos, obstacles, method=planning_method))
                all_paths.append(np.array(path_2[:path_index]))  # Save traversed path segments
                path_2 = new_path
                path_index = 0
            else:
                particle_pos += v * delta_t
                path_index += 1
                all_paths[-1] = np.append(all_paths[-1], [path_2[path_index]], axis=0)  # Update the latest path segment
        else:
            user_goal = goal_pos
            user_goal_path.append(user_goal)
            v = grad_U_pot(particle_pos, user_goal, obstacles, rho_0=rho0)
            v_magnitude = np.linalg.norm(v)
            v_direction = v / (v_magnitude + 1e-5)
            if np.linalg.norm(v_direction) == 0:
                v_direction = np.zeros(2)
            if v_magnitude > v_max:
                v = v_direction * v_max
            particle_pos += v * delta_t

        particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
        particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

        timestamp = time.time() - start_time
        collision = any(np.linalg.norm(particle_pos - obs['position']) < obs['radius'] for obs in obstacles)
        if collision==True:
            print(f"{collision}")
        # Determine success
        if collision:
            success = False

        data["timestamp"].append(timestamp)
        data["particle_position"].append(particle_pos.tolist())
        data["user_goal"].append(user_goal.tolist())
        data["velocity"].append(v.tolist())
        data["collision"].append(collision)
        data["success"].append(success)

        trajectory.append(particle_pos.copy())
        if np.linalg.norm(particle_pos - goal_pos) < 5:
            print("Particle reached goal")
            break

        # Draw obstacles
        for obstacle in obstacles:
            pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

        # Draw start and goal positions
        pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
        pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)

        # Draw particle's current position
        pygame.draw.circle(screen, BLUE, particle_pos.astype(int), 5)

        # Draw particle's trajectory
        for j in range(1, len(trajectory)):
            pygame.draw.line(screen, RED, trajectory[j - 1].astype(int), trajectory[j].astype(int), 1)

        # Draw all path segments
        for path in all_paths:
            for j in range(1, len(path)):
                pygame.draw.line(screen, [150, 150, 150], path[j - 1].astype(int), path[j].astype(int), 3)

        # Draw user goal path
        for j in range(1, len(user_goal_path)):
            pygame.draw.line(screen, BLUE, user_goal_path[j - 1].astype(int), user_goal_path[j].astype(int), 2)

        pygame.display.flip()
        pygame.time.delay(50)

    pygame.quit()

    # Define save path dynamically
    save_path = os.path.join("/Users/yuanzhengsun/Desktop/CBF_sim/CBF/APF_motion_planner", f"{i}.csv")
    print(f"Saving data to: {save_path}")

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {i}")

sys.exit()