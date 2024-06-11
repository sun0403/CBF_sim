import pygame
import numpy as np
import time
import sys

# 初始化APF算法常量
K_attr = 1.0
K_rep = 1.0
rho0 = 0.1

def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']

# 定义梯度函数
def grad_U_pot(x, x_goal, obs, rho_0):
    grad_U_attr = K_attr * (x_goal - x)
    grad_U_obs = np.zeros_like(x, dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1 / rho_x - 1 / rho_0) * (-1 / rho_x ** 2) * (x - ob['position']) / np.linalg.norm(x - ob['position'])
    return grad_U_attr + grad_U_obs

def find_path(x0, x_goal, rho_0, obs, alpha, max_iter=10000, tol=1e-3):
    x = x0  # 使用copy避免改变输入参数
    path = [x]
    for _ in range(max_iter):
        F = grad_U_pot(x, x_goal, obs, rho_0)
        x = x + alpha * F
        path.append(x)
        if np.linalg.norm(x - x_goal) < tol:
            break
    return np.array(path)

# 初始化Pygame
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
delta_t = 0.5
particle_speed = 50  # 调整速度
alpha = 0.01  # 力的缩放因子

# 生成随机障碍物
def generate_random_obstacles(num_obstacles, start_pos, goal_pos, field_size=3):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            position = np.random.rand(2) * field_size
            new_obstacle = {'position': position, 'radius': 0.5}
            if (np.linalg.norm(position - start_pos) > new_obstacle['radius'] and
                np.linalg.norm(position - goal_pos) > new_obstacle['radius'] and
                all(np.linalg.norm(position - obs['position']) > (new_obstacle['radius'] + obs['radius']) for obs in obstacles)):
                obstacles.append(new_obstacle)
                break
    return obstacles

# 生成障碍物并缩放到屏幕大小


# 起始和目标位置
start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
target_goal=np.array([50.0, 50.0])
particle_pos = np.array([50.0, 50.0])
velocity = np.array([0.0, 0.0])
num_obstacles = 3
obstacles =generate_random_obstacles(num_obstacles,start_pos,goal_pos)
for obs in obstacles:
    obs['position'] = obs['position'] * 100  # 缩放以适应屏幕
    obs['radius'] = obs['radius'] * 100 / 1.5  # 相应地缩放半径
# 碰撞检测函数
def check_collision(particle_pos, particle_radius, obstacles):
    for obstacle in obstacles:
        distance = np.linalg.norm(np.array(particle_pos) - np.array(obstacle['position']))
        if distance < particle_radius + obstacle['radius']+2:
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
    velocity = np.array([0.0, 0.0])
    # 更新目标位置
    if keys[pygame.K_LEFT]:
        velocity[0] -= particle_speed * delta_t
    if keys[pygame.K_RIGHT]:
        velocity[0] += particle_speed * delta_t
    if keys[pygame.K_UP]:
        velocity[1] -= particle_speed * delta_t
    if keys[pygame.K_DOWN]:
        velocity[1] += particle_speed * delta_t

     # 计算目标位置
    target_goal += velocity * delta_t
    # 确保目标位置保持在屏幕边界内
    target_goal[0] = np.clip(target_goal[0], 0, screen_width)
    target_goal[1] = np.clip(target_goal[1], 0, screen_height)

    print("Target Goal:", target_goal)

    # 碰撞检测和处理
    if check_collision(particle_pos, 5, obstacles):
        # 如果检测到碰撞，重新计算路径以绕过障碍物
        print("Collision detected! Recalculating path...")
        path = find_path(particle_pos, target_goal, rho0, obstacles,alpha)
        particle_pos = path[1] if len(path) > 1 else particle_pos
    else:
        path = find_path(particle_pos, target_goal, rho0, obstacles, alpha)
        particle_pos = path[1] if len(path) > 1 else particle_pos

    print("Particle Position:", particle_pos)  # 打印粒子位置

    # 确保粒子保持在屏幕边界内
    particle_pos[0] = np.clip(particle_pos[0], 0, screen_width)
    particle_pos[1] = np.clip(particle_pos[1], 0, screen_height)

    # 绘制模拟
    screen.fill(WHITE)

    # 绘制起始和目标位置
    pygame.draw.circle(screen, GREEN, start_pos, 10)
    pygame.draw.circle(screen, RED, goal_pos, 10)

    # 绘制障碍物
    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'], obstacle['radius'])

    # 绘制粒子
    pygame.draw.circle(screen, BLUE, particle_pos, 5)
    pygame.draw.circle(screen, BLUE, target_goal, 2)
    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
