import pygame
import numpy as np
import time
import sys

# 初始化APF算法常量
K_attr = 2.0
K_rep = 2.0
rho0 = 2.5

def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']

# 定义梯度函数
def grad_U_pot(x, x_goal, obs, rho_0):
    grad_U_attr = K_attr * (x_goal - x)
    grad_U_obs = np.zeros_like(x, dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1 / rho_x - 1 / rho_0) * (-1 / rho_x ** 2) * (x - ob['position']) / (np.linalg.norm(x - ob['position']) + 1e-6)
    return -grad_U_attr - grad_U_obs

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

# 粒子属性
delta_t = 0.1
particle_speed =  30 # 调整速度



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



# 起始和目标位置
start_pos = np.array([50.0, 50.0])
target_goal=np.array([450.0, 450.0])
particle_pos = np.array([50.0, 50.0])
velocity = np.array([0.0, 0.0])
num_obstacles = 10
obstacles =generate_random_obstacles(num_obstacles,start_pos,target_goal)

# 碰撞检测函数
def check_collision(particle_pos, particle_radius, obstacles):
    for obstacle in obstacles:
        distance = np.linalg.norm(np.array(particle_pos) - np.array(obstacle['position']))
        if distance < particle_radius + obstacle['radius']:
            return True
    return False
# 主循环
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    keys = pygame.key.get_pressed()
    if not check_collision(particle_pos, 5, obstacles):
        if keys[pygame.K_LEFT]:
            velocity[0] -= particle_speed * delta_t
        if keys[pygame.K_RIGHT]:
            velocity[0] += particle_speed * delta_t
        if keys[pygame.K_UP]:
            velocity[1] -= particle_speed * delta_t
        if keys[pygame.K_DOWN]:
            velocity[1] += particle_speed * delta_t
        particle_pos+=velocity*delta_t
        velocity1=velocity

    else:
        velocity = np.array([0.0, 0.0])
        F = grad_U_pot(particle_pos, target_goal,obstacles, rho0)
        print(F)
        particle_pos+=F*0.001
        velocity = np.array([0.0, 0.0])



    particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
    particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

    # 绘制模拟
    screen.fill(WHITE)

    # 绘制起始和目标位置
    pygame.draw.circle(screen, GREEN, start_pos, 10)
    #pygame.draw.circle(screen, RED, goal_pos, 10)

    # 绘制障碍物
    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'], obstacle['radius'])

    # 绘制粒子
    pygame.draw.circle(screen, BLUE, particle_pos, 5)
    pygame.draw.circle(screen, BLUE, target_goal, 2)
    pygame.display.flip()
    pygame.time.delay(100)

pygame.quit()
sys.exit()
