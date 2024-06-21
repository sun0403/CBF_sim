import pygame
import numpy as np
import sys
import cvxpy as cp


K=100.0
d_obs=50.0


def h(x, obstacles,d_obs):
    h = []
    for obs in obstacles:
        h.append(np.linalg.norm(x - obs['position']) - d_obs)
    return h


# define gradient of CBF
def grad_h(x, obstacles):
    grad_h = []
    for obs in obstacles:
        grad_h.append((x - obs['position']) / np.linalg.norm(x - obs['position']))
    return grad_h


#定义期望速度函数define function of desired volecity
def v_des(x, x_goal):
    return -K * (x - x_goal)



def qp_solver(x, x_goal, obstacles, alpha,d_obs):
    v = cp.Variable(2)
    v_desired = v_des(x, x_goal)

    h_values = h(x, obstacles,d_obs)
    grad_h_values = grad_h(x, obstacles)

    constraints = []
    for h_val, grad_h_val in zip(h_values, grad_h_values):
        constraints.append(grad_h_val @ v >= -alpha * h_val)

    objective = cp.Minimize(cp.sum_squares(v - v_desired))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS,verbose=True)

    return v.value

pygame.init()

(screen_width, screen_height) = 1000, 1000
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
target_goal = np.array([450.0, 450.0])
particle_pos = np.array([50.0, 50.0])
num_obstacles = 30
limlit=100
obstacles = generate_random_obstacles(num_obstacles, start_pos, goal_pos)
delta_t = 0.01
particle_speed = 200
alpha = 1.0




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
    if keys[pygame.K_LEFT]:
        left_pressed = True
        velocity[0] -= particle_speed * delta_t

    right_pressed = False
    if keys[pygame.K_RIGHT]:
        right_pressed = True
        velocity[0] += particle_speed * delta_t

    up_pressed = False
    if keys[pygame.K_UP]:
        up_pressed = True
        velocity[1] -= particle_speed * delta_t

    down_pressed = False
    if keys[pygame.K_DOWN]:
        down_pressed = True
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
    v = qp_solver(particle_pos,user_goal,obstacles,alpha,d_obs)
    particle_pos += v*0.05
    #Print the user_goal and particle_position
    print(f'Desired user goal: {user_goal}')
    print(f'Actual particle position: {particle_pos}')
    print(f'Desired user velocity: {velocity}')
    print(f'Actual velocity from v: {v}')

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
sys.exit()
