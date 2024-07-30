import time
import pygame
import numpy as np
import sys
import random
import motion_planner as mp
import pandas as pd

for i in range(10,20):
    np.random.seed(i)
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

    def generate_random_obstacles(num_obstacles, start_pos, goal_pos, d_obs,field_size=500):
        obstacles = []
        for _ in range(num_obstacles):
            while True:
                position = np.random.rand(2) * field_size
                new_obstacle = {'position': position, 'radius': 50}
                if (np.linalg.norm(position - start_pos) > (new_obstacle['radius'] + d_obs) and
                        np.linalg.norm(position - goal_pos) > (new_obstacle['radius'] + d_obs) and
                        all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in
                            obstacles)):
                    obstacles.append(new_obstacle)
                    break
        return obstacles

    def angle_between(v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    data = {
        "timestamp": [],
        "particle_position": [],
        "user_goal": [],
        "velocity": [],
        "collision": [],
    }
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
    d_obs = 20
    obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos,
                                          d_obs, screen_height)
    K_att=30.0
    K_rep=60.0
    delta=1.0
    v_max = 500.0
    angle_threshold = np.pi / 2  # Set angle threshold to 60 degrees
    delta_t = 0.01
    running = True

    rho_0=10.0

    # Initialize the motion planner
    planner = mp.MotionPlanner(grid_size=500, grid_step=5)
    path_2 = np.array(planner.select_random_planner(start_pos, goal_pos, obstacles))

    path_index = 0
    trajectory = []
    all_paths = [path_2[:path_index + 1].tolist()]  # Initialize to include start point
    user_goal_path = []

    previous_error = 0
    total_error = 0

    k_p = 100.0
    k_d = 5.0
    k_i = 0.5

    # Main loop
    start_time = time.time()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(WHITE)

        if path_index < len(path_2)-1:
            # PID controller
            error_pos = particle_pos - path_2[path_index + 1]
            error_delta = error_pos - previous_error
            total_error += error_pos  # Accumulated error
            previous_error = error_pos
            u = k_p * error_pos + k_d * error_delta + k_i * total_error
            user_goal = particle_pos - u * delta_t
            user_goal_path.append(user_goal)

            v = v_star(particle_pos, user_goal, obstacles, alpha=1.0,
                       delta=delta, rho_0=rho_0)
            v_direction = v / np.linalg.norm(v)
            if np.linalg.norm(v_direction) == 0:
                v_direction = np.zeros(2)
            path_direction = (user_goal - particle_pos) / np.linalg.norm(user_goal - particle_pos)
            angle_diff = angle_between(v_direction, path_direction)
            if angle_diff > angle_threshold:
                print("Particle1")
                new_path = np.array(planner.rrt(particle_pos, goal_pos, obstacles))
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
            v = v_star(particle_pos, user_goal, obstacles, alpha=1.0,
                       delta=delta, rho_0=rho_0)
            particle_pos += v * delta_t

        particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
        particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

        timestamp = time.time() - start_time
        collision = any(np.linalg.norm(particle_pos - obs['position']) < obs['radius'] for obs in obstacles)

        data["timestamp"].append(timestamp)
        data["particle_position"].append(particle_pos.tolist())
        data["user_goal"].append(user_goal.tolist())
        data["velocity"].append(v.tolist())
        data["collision"].append(collision)

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
                pygame.draw.line(screen, [150, 150, 150], np.array(path[j - 1]).astype(int), np.array(path[j]).astype(int), 3)

        # Draw user goal path
        for j in range(1, len(user_goal_path)):
            pygame.draw.line(screen, BLUE, user_goal_path[j - 1].astype(int), user_goal_path[j].astype(int), 1)

        pygame.display.flip()
        pygame.time.delay(50)

    pygame.quit()
    df = pd.DataFrame(data)
    df.to_csv(f"/home/sun/CBF_sim/APF+CBF_motion_planner/{i}.csv", index=False)
    print(f"Data saved to {i}.csv")
sys.exit()

