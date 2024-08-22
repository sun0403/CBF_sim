import pandas as pd
import pygame
import numpy as np
import time
import sys

np.random.seed(76)


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

pygame.init()

# 屏幕大小
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("粒子控制模拟")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


field_size=500
num_obstacles = 10
limlit=500
delta_t = 0.01
particle_speed = 500
K_attr = 100.0
K_rep = 10000.0
rho0 = 50.0
for i in range(10):
    start_pos = np.array([50.0, 50.0])
    target_goal=np.array([450.0, 450.0])
    particle_pos = np.array([50.0, 50.0])
    velocity = np.array([0.0, 0.0])

    data = {
        "delta_t": [],
        "time_steps": [],
        "timestamp": [],
        "particle_position": [],
        "user_goal": [],
        "velocity": [],
        "collision": [],
        "key_pressed": [],
        "success": [],
    }

    obstacles =generate_random_obstacles(num_obstacles,start_pos,target_goal,d_obs=60,field_size=field_size)
    # Add boundaries as obstacles
    boundary_thickness =2
    boundaries = [
        {'position': np.array([screen_width / 2, boundary_thickness / 2]), 'radius': boundary_thickness},
        {'position': np.array([screen_width / 2, screen_height - boundary_thickness / 2]), 'radius': boundary_thickness},
        {'position': np.array([boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness},
        {'position': np.array([screen_width - boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness}
    ]

    # Add boundaries to the list of obstacles
    obstacles.extend(boundaries)


    running = True
    user_goal = np.array([0.0,0.0])

    success = True
    start_time = time.time()
    time_steps = 0

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

        any_key_pressed = left_pressed or right_pressed or up_pressed or down_pressed

        # If no key is pressed, set velocity to zero
        if not any_key_pressed:
            velocity *= 0.0
        #update the postion
        user_goal=particle_pos+delta_t*velocity
        F = grad_U_pot(particle_pos, user_goal, obstacles, rho0)
        F = np.clip(F, -500, 500)
        particle_pos += F * delta_t

        print(f'Desired user goal: {user_goal}')
        print(f'Actual particle position: {particle_pos}')
        print(f'Desired user velocity: {velocity}')
        print(f'Actual velocity from F: {F}')

        if np.linalg.norm(particle_pos-target_goal)<5:
            break
        #collect the data
        timestamp = time.time() - start_time
        collision = any(np.linalg.norm(particle_pos - obs['position']) < obs['radius'] for obs in obstacles)

        data["delta_t"].append(delta_t)
        data["time_steps"].append(time_steps)
        data["timestamp"].append(timestamp)
        data["particle_position"].append(particle_pos.tolist())
        data["user_goal"].append(user_goal.tolist())
        data["velocity"].append(velocity.tolist())
        data["collision"].append(collision)
        data["key_pressed"].append(key_pressed)
        data["success"].append(success)

        particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
        particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

        screen.fill(WHITE)
        pygame.draw.circle(screen, GREEN, start_pos, 10)
        #pygame.draw.circle(screen, RED, goal_pos, 10)

        # draw the obstacles
        for obstacle in obstacles:
            pygame.draw.circle(screen, BLACK, obstacle['position'], obstacle['radius'])

        # draw the position
        pygame.draw.circle(screen, BLUE, particle_pos, 5)
        pygame.draw.circle(screen, BLUE, target_goal, 2)
        pygame.display.flip()
        pygame.time.delay(50)
        time_steps += delta_t

    df = pd.DataFrame(data)
    #path = f"./l/APF/{i}.csv"
    path=f"./APF_csv/{i}.csv"
    df.to_csv(path, index=False)
    print(f"Data saved to {i}.csv")


pygame().quit()
