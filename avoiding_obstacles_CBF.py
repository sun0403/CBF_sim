import pygame
import time
import sys
import numpy as np
import CBF
# 初始化APF算法相关的常量
K_attr = 1.0
K_rep = 1.0
rho0 = 0.1



# 初始化pygame
pygame.init()

# 屏幕大小
screen_width, screen_height = 500, 500
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Particle Control Simulation")

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# 粒子属性
particle_vel = [0.0, 0.0]
delta_t = 0.1
particle_speed = 10  # 调整速度

# 生成随机障碍物
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

# 生成障碍物并缩放到屏幕大小
num_obstacles = 10
obstacles = generate_random_obstacles(num_obstacles)
for obs in obstacles:
    obs['position'] = obs['position'] * 100  # 缩放以适应屏幕
    obs['radius'] = obs['radius'] * 100 / 2.5  # 相应地缩放半径

# 起始和目标位置
start_pos = np.array([50.0, 50.0])
goal_pos = np.array([450.0, 450.0])
particle_pos = start_pos.copy()

# 碰撞检测函数
def check_collision(particle_pos, particle_radius, obstacles):
    for obstacle in obstacles:
        distance = np.linalg.norm(np.array(particle_pos) - np.array(obstacle['position']))
        if distance < particle_radius + obstacle['radius']+1:
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

    # 调用速度控制函数
    keys = pygame.key.get_pressed()

    # 计算期望目标位置
    desired_goal = particle_pos.copy()
    if keys[pygame.K_LEFT]:
        desired_goal[0] -= particle_speed * delta_t
    if keys[pygame.K_RIGHT]:
        desired_goal[0] += particle_speed * delta_t
    if keys[pygame.K_UP]:
        desired_goal[1] -= particle_speed * delta_t
    if keys[pygame.K_DOWN]:
        desired_goal[1] += particle_speed * delta_t
#
    # 确保粒子保持在屏幕边界内
    desired_goal[0] = np.clip(desired_goal[0], 0, screen_width)
    desired_goal[1] = np.clip(desired_goal[1], 0, screen_height)

    # 使用APF算法计算路径
    if check_collision(desired_goal, 5, obstacles):
        path, _, _ = CBF.find_path_qp(particle_pos,desired_goal,obstacles,alpha=2)
        if len(path) > 1:
            particle_pos = np.array(path[1])  # 移动到路径的下一个位置
    else:
        particle_pos = desired_goal  # 没有碰撞时按期望目标位置移动

    # 绘制
    screen.fill(WHITE)

    # 绘制起点和终点
    pygame.draw.circle(screen, GREEN, start_pos.astype(int), 10)
    pygame.draw.circle(screen, RED, goal_pos.astype(int), 10)

    # 绘制障碍物
    for obstacle in obstacles:
        pygame.draw.circle(screen, BLACK, obstacle['position'].astype(int), int(obstacle['radius']))

    # 绘制粒子
    pygame.draw.circle(screen, BLUE, (int(particle_pos[0]), int(particle_pos[1])), 5)

    pygame.display.flip()
    pygame.time.delay(50)

pygame.quit()
sys.exit()
