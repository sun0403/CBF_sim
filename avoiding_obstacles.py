import pygame
import random
import sys
import numpy as np
import APF

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
velocity_control = True
delta_t = 0.01
particle_speed = 50  # Adjust the speed here
K_attr = 1.0
K_rep = 1.0
# Generate random obstacles
def generate_random_obstacles(num_obstacles, field_size=5):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            position = np.random.rand(2) * field_size
            new_obstacle = {'position': position, 'radius': 0.5}
            if all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in obstacles):
                obstacles.append(new_obstacle)
                break
    return obstacles
def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']

#定义梯度函数define gradient function
def grad_U_pot(x,x_goal,obs,rho_0):
    grad_U_attr = K_attr * (x - x_goal)
    grad_U_obs = np.zeros_like(x,dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1/rho_x - 1/rho_0) * (-1/rho_x**2) * (x - ob['position']) / np.linalg.norm(x - ob['position'])
        else:
            grad_U_obs += 0
    return -grad_U_attr - grad_U_obs
# Generate obstacles and scale to screen size
num_obstacles = 20
obstacles = generate_random_obstacles(num_obstacles)

for obs in obstacles:
    obs['position'] = obs['position'] * 100  # scale to fit screen
    obs['radius'] = obs['radius'] * 100 / 5  # scale radius accordingly

# Start and goal positions
start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos = start_pos.copy()

# Collision detection function
def check_collision(particle_pos, particle_radius, obstacles,rho0):
    for obstacle in obstacles:
        distance = np.linalg.norm(np.array(particle_pos) - np.array(obstacle['position']))
        if distance < particle_radius + obstacle['radius']+5:
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
            if event.key == pygame.K_v:
                velocity_control = not velocity_control

    # Call velocity control function
    keys = pygame.key.get_pressed()

    if velocity_control:
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
    else:

        particle_acc=[0,0]
        if keys[pygame.K_LEFT]:

            particle_acc[0] -= particle_speed
        if keys[pygame.K_RIGHT]:

            particle_acc[0] += particle_speed
        if keys[pygame.K_UP]:

            particle_acc[1] -= particle_speed
        if keys[pygame.K_DOWN]:

            particle_acc[1] += particle_speed

        particle_vel[0] += particle_acc[0] * delta_t
        particle_vel[1] += particle_acc[1] * delta_t
        particle_pos[0] += particle_vel[0] * delta_t
        particle_pos[1] += particle_vel[1] * delta_t
    obs=obstacles
    rho0=2
    # Collision detection
    if check_collision(particle_pos, 5, obstacles,rho0):
        # Call APF to find a new path
        F = grad_U_pot(particle_pos, goal_pos, obs, rho0)
        particle_pos = particle_pos+0.001*F


    if particle_pos[0] < 0:
        particle_pos[0] = 0
        particle_vel[0] = 0
    elif particle_pos[0] > screen_width:
        particle_pos[0] = screen_width
        particle_vel[0] = 0

    if particle_pos[1] < 0:
        particle_pos[1] = 0
        particle_vel[1] = 0
    elif particle_pos[1] > screen_height:
        particle_pos[1] = screen_height
        particle_vel[1] = 0

    # Draw
    screen.fill(WHITE)

    # Draw start and goal positions
    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)

    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    # Draw particle
    pygame.draw.circle(screen, BLUE, (int(particle_pos[0]), int(particle_pos[1])), 5)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
