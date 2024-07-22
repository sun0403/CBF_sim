import time
import pygame
import numpy as np
import sys
import cvxpy as cp
import motion_planner as mp


# Function definitions
def h(x, obstacles, d_obs):
    """Calculate the distance from the particle to each obstacle."""
    h = []
    for obs in obstacles:
        h.append(np.linalg.norm(x - obs['position']) - d_obs)
    return h


def grad_h(x, obstacles):
    """Calculate the gradient of the distance function."""
    grad_h = []
    for obs in obstacles:
        grad_h.append((x - obs['position']) / np.linalg.norm(x - obs['position']))
    return grad_h


def v_des(x, x_goal, K):
    """Calculate the desired velocity."""
    return -K * (x - x_goal)


def qp_solver(x, x_goal, obstacles, alpha, d_obs, v_max, K):
    """Solve the QP problem to determine the optimal velocity."""
    v = cp.Variable(2)
    v_desired = v_des(x, x_goal, K)

    h_values = h(x, obstacles, d_obs)
    grad_h_values = grad_h(x, obstacles)

    constraints = []
    for h_val, grad_h_val in zip(h_values, grad_h_values):
        constraints.append(grad_h_val @ v >= -alpha * h_val)

    constraints.append(cp.norm(v, 2) <= v_max)

    objective = cp.Minimize(cp.sum_squares(v - v_desired))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    return v.value


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


# Initialization
pygame.init()

screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Simulation Control")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos = np.array([50.0, 50.0])
num_obstacles = 10
particle_speed = 200
d_obs = 10
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos, d_obs, screen_height)

K = 200.0
v_max = 500.0
angle_threshold = np.pi / 3  # Set angle threshold to 60 degrees
delta_t = 0.01
running = True
d = 50

# Initialize the motion planner
planner = mp.MotionPlanner(grid_size=500, grid_step=5)
path_2 = np.array(planner.select_random_planner(start_pos, goal_pos, obstacles))

path_index = 0
trajectory = []
all_paths = [path_2[:path_index + 1].tolist()]  # Initialize to include start point
user_goal_path = []

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
        # PID controller
        error_pos = particle_pos - path_2[path_index + 1]
        error_delta = error_pos - previous_error
        total_error += error_pos  # Accumulated error
        previous_error = error_pos
        u = k_p * error_pos + k_d * error_delta + k_i * total_error
        user_goal = particle_pos - u * delta_t
        user_goal_path.append(user_goal)

        v = qp_solver(particle_pos, user_goal, obstacles, alpha=10.0,
                      v_max=v_max, K=K, d_obs=d)
        v_direction = v / np.linalg.norm(v)
        if np.linalg.norm(v_direction) == 0:
            v_direction = np.zeros(2)
        path_direction = (user_goal - particle_pos) / np.linalg.norm(user_goal - particle_pos)
        angle_diff = angle_between(v_direction, path_direction)
        if angle_diff > angle_threshold:
            new_path = np.array(planner.select_random_planner(particle_pos, goal_pos, obstacles))
            all_paths.append(path_2[:path_index].tolist())  # Save traversed path segments
            path_2 = new_path
            path_index = 0
        else:
            particle_pos += v * delta_t
            path_index += 1
            all_paths[-1].append(path_2[path_index].tolist())  # Update the latest path segment
    else:
        user_goal = goal_pos
        user_goal_path.append(user_goal)
        v = qp_solver(particle_pos, user_goal, obstacles, alpha=10.0,
                      v_max=v_max, K=K, d_obs=d)
        particle_pos += v * delta_t

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

    # Draw particle's current position
    pygame.draw.circle(screen, BLUE, particle_pos.astype(int), 5)

    # Draw particle's trajectory
    for j in range(1, len(trajectory)):
        pygame.draw.line(screen, RED, trajectory[j - 1].astype(int), trajectory[j].astype(int), 1)

    # Draw all path segments
    for path in all_paths:
        for j in range(1, len(path)):
            pygame.draw.line(screen, [150, 150, 150], np.array(path[j - 1]).astype(int), np.array(path[j]).astype(int), 3)

    # Draw user goal path
    for j in range(1, len(user_goal_path)):
        pygame.draw.line(screen, BLUE, user_goal_path[j - 1].astype(int), user_goal_path[j].astype(int), 1)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
