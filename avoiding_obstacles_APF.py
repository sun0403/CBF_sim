import pygame
import sys
import numpy as np
import time

# Initialize APF algorithm constants
K_attr = 1.0
K_rep = 1.0
rho0 = 0.25


def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']


# Define the gradient function
def grad_U_pot(x, x_goal, obs, rho_0):
    grad_U_attr = K_attr * (x_goal - x)
    grad_U_obs = np.zeros_like(x, dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1 / rho_x - 1 / rho_0) * (-1 / rho_x ** 2) * (x - ob['position']) / np.linalg.norm(
                x - ob['position'])
    return grad_U_attr + grad_U_obs


def find_path(x0, x_goal, rho_0, obs, alpha=0.001, max_iter=10000, tol=1e-3):
    x = x0.copy()
    path = [x]
    times = [0]
    start_time = time.time()
    for _ in range(max_iter):
        F = grad_U_pot(x, x_goal, obs, rho_0)
        x = x + alpha * F
        times.append(time.time() - start_time)
        path.append(x.copy())
        if np.linalg.norm(x - x_goal) < tol:
            break
    final_time = time.time() - start_time
    return np.array(path), final_time, times


# Initialize pygame
pygame.init()

# Screen size
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Control Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Particle properties
particle_vel = [0.0, 0.0]
delta_t = 0.1
particle_speed = 20  # Adjust the speed


# Generate random obstacles
def generate_random_obstacles(num_obstacles, field_size=5):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            position = np.random.rand(2) * field_size
            new_obstacle = {'position': position, 'radius': 0.5}
            if all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in
                   obstacles):
                obstacles.append(new_obstacle)
                break
    return obstacles


# Generate obstacles and scale to screen size
num_obstacles = 10
obstacles = generate_random_obstacles(num_obstacles)
for obs in obstacles:
    obs['position'] = obs['position'] * 100  # Scale to fit screen
    obs['radius'] = obs['radius'] * 100 / 1.5  # Scale radius accordingly

# Start and goal positions
start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos = start_pos.copy()


# Collision detection function
def check_collision(particle_pos, particle_radius, obstacles):
    for obstacle in obstacles:
        distance = np.linalg.norm(np.array(particle_pos) - np.array(obstacle['position']))
        if distance < particle_radius + obstacle['radius']:
            return True
    return False


# Main loop
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    keys = pygame.key.get_pressed()

    # Reset velocity if no key is pressed
    particle_vel = [0, 0]
    if keys[pygame.K_LEFT]:
        particle_vel[0] -= particle_speed
    if keys[pygame.K_RIGHT]:
        particle_vel[0] += particle_speed
    if keys[pygame.K_UP]:
        particle_vel[1] -= particle_speed
    if keys[pygame.K_DOWN]:
        particle_vel[1] += particle_speed

    # Update particle position
    particle_pos[0] += particle_vel[0] * delta_t
    particle_pos[1] += particle_vel[1] * delta_t

    # Collision detection and handling
    if check_collision(particle_pos, 5, obstacles):
        # Call APF to find a new path
        path, _, _ = find_path(particle_pos, goal_pos, rho_0=0.5, obs=obstacles)
        particle_pos = np.array(path[0])

    # Ensure the particle stays within the screen boundaries
    particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
    particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

    # Draw the simulation
    screen.fill(WHITE)

    # Draw start and goal positions
    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    # Draw particle
    pygame.draw.circle(screen, BLUE, particle_pos.astype(int), 5)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
