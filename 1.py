import time
import pygame
import numpy as np

from motion_planner import MotionPlanner  # Import the MotionPlanner class

np.random.seed(3)
pygame.init()

(screen_width, screen_height) = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Control Simulation")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

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
particle_pos_a_star = np.array([50.0, 50.0])
particle_pos_bfs = np.array([50.0, 50.0])
particle_pos_rrt = np.array([50.0, 50.0])
num_obstacles = 20

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

data_a_star = {
    "timestamp": [],
    "particle_position": [],
    "user_goal": [],
    "velocity": [],
    "collision": []
}

data_bfs = {
    "timestamp": [],
    "particle_position": [],
    "user_goal": [],
    "velocity": [],
    "collision": []
}

data_rrt = {
    "timestamp": [],
    "particle_position": [],
    "user_goal": [],
    "velocity": [],
    "collision": []
}

start_time = time.time()

planner = MotionPlanner(grid_size=500, grid_step=10)
path_a_star = planner.a_star(start_pos, goal_pos, obstacles)
path_bfs = planner.bfs(start_pos, goal_pos, obstacles)
path_rrt = planner.rrt(start_pos, goal_pos, obstacles)
path_index_a_star = 0
path_index_bfs = 0
path_index_rrt = 0
trajectory_a_star = []
trajectory_bfs = []
trajectory_rrt = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    # A* path following
    if path_index_a_star < len(path_a_star):
        a_star_goal = np.array(path_a_star[path_index_a_star])
        direction_a_star = a_star_goal - particle_pos_a_star
        distance_a_star = np.linalg.norm(direction_a_star)
        if distance_a_star < 5:
            path_index_a_star += 1
        else:
            direction_a_star = direction_a_star / distance_a_star * particle_speed * delta_t
            particle_pos_a_star += direction_a_star
        user_goal_a_star = a_star_goal
    else:
        user_goal_a_star = goal_pos

    # BFS path following
    if path_index_bfs < len(path_bfs):
        bfs_goal = np.array(path_bfs[path_index_bfs])
        direction_bfs = bfs_goal - particle_pos_bfs
        distance_bfs = np.linalg.norm(direction_bfs)
        if distance_bfs < 5:
            path_index_bfs += 1
        else:
            direction_bfs = direction_bfs / distance_bfs * particle_speed * delta_t
            particle_pos_bfs += direction_bfs
        user_goal_bfs = bfs_goal
    else:
        user_goal_bfs = goal_pos

    # RRT path following
    if path_index_rrt < len(path_rrt):
        rrt_goal = np.array(path_rrt[path_index_rrt])
        direction_rrt = rrt_goal - particle_pos_rrt
        distance_rrt = np.linalg.norm(direction_rrt)
        if distance_rrt < 5:
            path_index_rrt += 1
        else:
            direction_rrt = direction_rrt / distance_rrt * particle_speed * delta_t
            particle_pos_rrt += direction_rrt
        user_goal_rrt = rrt_goal
    else:
        user_goal_rrt = goal_pos

    timestamp = time.time() - start_time
    collision_a_star = any(np.linalg.norm(particle_pos_a_star - obs['position']) < obs['radius'] for obs in obstacles)
    collision_bfs = any(np.linalg.norm(particle_pos_bfs - obs['position']) < obs['radius'] for obs in obstacles)
    collision_rrt = any(np.linalg.norm(particle_pos_rrt - obs['position']) < obs['radius'] for obs in obstacles)

    data_a_star["timestamp"].append(timestamp)
    data_a_star["particle_position"].append(particle_pos_a_star.tolist())
    data_a_star["user_goal"].append(user_goal_a_star.tolist())
    data_a_star["velocity"].append(direction_a_star.tolist() if path_index_a_star < len(path_a_star) else [0, 0])
    data_a_star["collision"].append(collision_a_star)

    data_bfs["timestamp"].append(timestamp)
    data_bfs["particle_position"].append(particle_pos_bfs.tolist())
    data_bfs["user_goal"].append(user_goal_bfs.tolist())
    data_bfs["velocity"].append(direction_bfs.tolist() if path_index_bfs < len(path_bfs) else [0, 0])
    data_bfs["collision"].append(collision_bfs)

    data_rrt["timestamp"].append(timestamp)
    data_rrt["particle_position"].append(particle_pos_rrt.tolist())
    data_rrt["user_goal"].append(user_goal_rrt.tolist())
    data_rrt["velocity"].append(direction_rrt.tolist() if path_index_rrt < len(path_rrt) else [0, 0])
    data_rrt["collision"].append(collision_rrt)

    trajectory_a_star.append(particle_pos_a_star.copy())
    trajectory_bfs.append(particle_pos_bfs.copy())
    trajectory_rrt.append(particle_pos_rrt.copy())



    if np.linalg.norm(particle_pos_a_star - goal_pos) < 5 and np.linalg.norm(particle_pos_bfs - goal_pos) < 5 and np.linalg.norm(particle_pos_rrt - goal_pos) < 5:
        print("Particle reached goal")
        running = False

    particle_pos_a_star[0] = np.clip(particle_pos_a_star[0], 0, screen_width)
    particle_pos_a_star[1] = np.clip(particle_pos_a_star[1], 0, screen_height)
    particle_pos_bfs[0] = np.clip(particle_pos_bfs[0], 0, screen_width)
    particle_pos_bfs[1] = np.clip(particle_pos_bfs[1], 0, screen_height)
    particle_pos_rrt[0] = np.clip(particle_pos_rrt[0], 0, screen_width)
    particle_pos_rrt[1] = np.clip(particle_pos_rrt[1], 0, screen_height)

    screen.fill(WHITE)

    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)

    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    pygame.draw.circle(screen, BLUE, particle_pos_a_star.astype(int), 5)
    pygame.draw.circle(screen, BLACK, particle_pos_bfs.astype(int), 5)
    pygame.draw.circle(screen, RED, particle_pos_rrt.astype(int), 5)

    # Draw trajectories
    for j in range(1, len(trajectory_a_star)):
        pygame.draw.line(screen, BLUE, trajectory_a_star[j-1], trajectory_a_star[j], 2)
    for j in range(1, len(trajectory_bfs)):
        pygame.draw.line(screen, BLACK, trajectory_bfs[j-1], trajectory_bfs[j], 2)
    for j in range(1, len(trajectory_rrt)):
        pygame.draw.line(screen, RED, trajectory_rrt[j-1], trajectory_rrt[j], 2)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()