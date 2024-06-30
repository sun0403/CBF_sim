import time
import pygame
import numpy as np
import sys
import cvxpy as cp
import pandas as pd

K = 100.0
d_obs = 50.0

def h(x, obstacles, d_obs):
    h = []
    for obs in obstacles:
        h.append(np.linalg.norm(x - obs['position']) - d_obs)
    return h

def grad_h(x, obstacles):
    grad_h = []
    for obs in obstacles:
        grad_h.append((x - obs['position']) / np.linalg.norm(x - obs['position']))
    return grad_h

def v_des(x, x_goal):
    return -K * (x - x_goal)

def qp_solver(x, x_goal, obstacles, alpha, d_obs):
    v = cp.Variable(2)
    v_desired = v_des(x, x_goal)

    h_values = h(x, obstacles, d_obs)
    grad_h_values = grad_h(x, obstacles)

    constraints = []
    for h_val, grad_h_val in zip(h_values, grad_h_values):
        constraints.append(grad_h_val @ v >= -alpha * h_val)

    objective = cp.Minimize(cp.sum_squares(v - v_desired))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    return v.value

pygame.init()

(screen_width, screen_height) = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Control Simulation")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

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

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
target_goal = np.array([450.0, 450.0])
particle_pos = np.array([50.0, 50.0])
num_obstacles = 10
limlit = 100
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos)
boundary_thickness =10
boundaries = [
    {'position': np.array([screen_width / 2, boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width / 2, screen_height - boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width - boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness}
]
delta_t = 0.01
particle_speed = 200
alpha = 1.0

velocity = np.array([0.0, 0.0])
running = True
user_goal = np.array([0.0, 0.0])


data = {
    "timestamp": [],
    "particle_position": [],
    "user_goal": [],
    "velocity": [],
    "collision": [],
    "key_pressed": []
}

start_time = time.time()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    keys = pygame.key.get_pressed()
    key_pressed = None
    left_pressed = False
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

    any_key_pressed = left_pressed or right_pressed or up_pressed or down_pressed

    if not any_key_pressed:
        velocity *= 0.0

    user_goal = particle_pos + velocity * delta_t
    v = qp_solver(particle_pos, user_goal, obstacles, alpha, d_obs)
    particle_pos += v * 0.05
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
    pygame.time.delay(50)

pygame.quit()

df = pd.DataFrame(data)
df.to_csv("CBF10.csv", index=False)
print("Data saved to CBF.csv")


'''task_completion_time = timestamp
human_fatigue = len([k for k in data["key_pressed"] if k is not None])
smoothness = np.mean([np.linalg.norm(v) for v in data["velocity"]])
collisions = sum(data["collision"])


print(f"Task completion time: {task_completion_time} seconds")
print(f"Human fatigue (key presses): {human_fatigue}")
print(f"Smoothness of motions: {smoothness}")
print(f"Number of collisions: {collisions}")'''

sys.exit()
