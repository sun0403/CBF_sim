import pygame
import numpy as np
import sys
import time
import pandas as pd

K_att = 9.99
K_rep = 9.99
delta = 0.001
rho_0=20


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

pygame.init()

screen_width, screen_height = 1000, 1000
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Control Simulation")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)



def generate_random_obstacles(num_obstacles, start_pos, goal_pos, field_size=1000):
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

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
target_goal = np.array([950.0, 950.0])
particle_pos = np.array([50.0, 50.0])
num_obstacles = 30
limlit=100
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos)
boundary_thickness =2
boundaries = [
    {'position': np.array([screen_width / 2, boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width / 2, screen_height - boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width - boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness}
]
delta_t = 0.05
particle_speed = 100
alpha = 0.5

data = {
    "timestamp": [],
    "particle_position": [],
    "user_goal": [],
    "velocity": [],
    "collision": [],
    "key_pressed": []
}

start_time = time.time()




velocity = np.array([0.0, 0.0])
running = True
user_goal = np.array([0.0,0.0])
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    keys = pygame.key.get_pressed()
    left_pressed = False
    key_pressed = None
    if keys[pygame.K_LEFT]:
        left_pressed = True
        key_pressed = "LEFT"
        velocity[0] -= particle_speed * delta_t

    right_pressed = False
    if keys[pygame.K_RIGHT]:
        right_pressed = True
        key_pressed = "RIGHT"
        velocity[0] += particle_speed * delta_t

    up_pressed = False
    if keys[pygame.K_UP]:
        up_pressed = True
        key_pressed = "UP"
        velocity[1] -= particle_speed * delta_t

    down_pressed = False
    if keys[pygame.K_DOWN]:
        down_pressed = True
        key_pressed = "DOWN"
        velocity[1] += particle_speed * delta_t

    if velocity[0] > limlit:
        velocity[0] = limlit
    if velocity[0] < -limlit:
        velocity[0] = -limlit
    if velocity[1] > limlit:
        velocity[1] = limlit
    if velocity[1] < -limlit:
        velocity[1] = -limlit

    # Check if any key is pressed
    any_key_pressed = left_pressed or right_pressed or up_pressed or down_pressed

    # If no key is pressed, set velocity to zero
    if not any_key_pressed:
        velocity *= 0.0



    user_goal = particle_pos + velocity * delta_t
    v = v_star(particle_pos, user_goal, obstacles, alpha, delta=0.001, rho_0=rho_0)
    particle_pos += v*0.02


    timestamp = time.time() - start_time
    collision = any(np.linalg.norm(particle_pos - obs['position']) < obs['radius'] for obs in obstacles)

    data["timestamp"].append(timestamp)
    data["particle_position"].append(particle_pos.tolist())
    data["user_goal"].append(user_goal.tolist())
    data["velocity"].append(velocity.tolist())
    data["collision"].append(collision)
    data["key_pressed"].append(key_pressed)

    print(f'Desired user goal: {user_goal}')
    print(f'Actual particle position: {particle_pos}')
    print(f'Desired user velocity: {velocity}')
    print(f'Actual velocity from v: {v}')

    if np.linalg.norm(particle_pos - target_goal) < 5:
        print("Particle reached goal")
        break

    particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
    particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

    screen.fill(WHITE)

    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)

    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    pygame.draw.circle(screen, BLUE, particle_pos.astype(int), 5)
    pygame.draw.circle(screen, BLUE, target_goal.astype(int), 2)
    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()

df=pd.DataFrame(data)
df.to_csv("CBF+APF1.csv", index=False)
print("Data saved to CBF+APF3.csv")
sys.exit()
