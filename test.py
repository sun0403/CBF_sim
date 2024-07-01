import time
import pygame
import numpy as np
import sys
import pandas as pd
import motion_planner  # Assuming this is your custom module containing a_star function

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
                    all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in
                        obstacles)):
                obstacles.append(new_obstacle)
                break
    return obstacles

start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos = np.array([50.0, 50.0])
num_obstacles = 10
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos)
boundary_thickness = 10
boundaries = [
    {'position': np.array([screen_width / 2, boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width / 2, screen_height - boundary_thickness / 2]), 'radius': boundary_thickness},
    {'position': np.array([boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness},
    {'position': np.array([screen_width - boundary_thickness / 2, screen_height / 2]), 'radius': boundary_thickness}
]

delta_t = 0.01
particle_speed = 200
running = True

data = {
    "timestamp": [],
    "particle_position": [],
    "user_goal": [],
    "velocity": [],
    "collision": []
}

start_time = time.time()

a_star_path = motion_planner.a_star(start_pos, goal_pos, obstacles)
path_index = 0
trajectory = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    if path_index < len(a_star_path):
        a_star_goal = np.array(a_star_path[path_index])
        direction = a_star_goal - particle_pos
        distance = np.linalg.norm(direction)
        if distance < 5:
            path_index += 1
        else:
            direction = direction / distance * particle_speed * delta_t
            particle_pos += direction
        user_goal = a_star_goal
    else:
        user_goal = goal_pos

    timestamp = time.time() - start_time
    collision = any(np.linalg.norm(particle_pos - obs['position']) < obs['radius'] for obs in obstacles)

    data["timestamp"].append(timestamp)
    data["particle_position"].append(particle_pos.tolist())
    data["user_goal"].append(user_goal.tolist())
    data["velocity"].append(direction.tolist())
    data["collision"].append(collision)
    trajectory.append(particle_pos.copy())

    print(f'Desired user goal: {user_goal}')
    print(f'Actual particle position: {particle_pos}')
    print(f'Actual velocity: {direction}')

    if np.linalg.norm(particle_pos - goal_pos) < 5:
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

    # Draw trajectory
    for i in range(1, len(trajectory)):
        pygame.draw.line(screen, BLUE, trajectory[i-1], trajectory[i], 2)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()

df = pd.DataFrame(data)
df.to_csv("AStar_Path.csv", index=False)
print("Data saved to AStar_Path.csv")

sys.exit()
