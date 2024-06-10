import matplotlib.pyplot as plt
import numpy as np
import time
##
# 定义初始位置和目标位置 set start and goal
x0 = np.array([0.0, 0.0])
x_goal = np.array([3.0, 5.0])
#定义参数 set parameter
K_attr = 1.0
K_rep = 1.0
# 定义不同的ρ0和障碍 set rho0 and obstacles
rho_01 = 1.0
rho_02 = 0.5
rho_03 = 0.25
rho_04 = 0.1
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
num=5
obstacles =generate_random_obstacles(num)
'''[
    {'position': np.array([1.0, 2.0]), 'radius': 0.5},
{'position': np.array([2.5, 3.0]), 'radius': 0.5}
]'''
#定义rho0
def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']

#定义梯度函数define gradient function
def grad_U_pot(x,x_goal,obs,rho_0):
    grad_U_attr = K_attr * (x - x_goal)
    grad_U_obs = np.zeros_like(x,dtype=float)
    for ob in obs:
        rho_x = rho(x, ob)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1/rho_x - 1/rho_0) * (-1/rho_x**2) * (x - ob['position']) / np.linalg.norm(x - ob['position'])
        else:
            grad_U_obs += 0
    return -grad_U_attr - grad_U_obs
#定义寻路函数
def find_path(x0, x_goal,rho_0,obs,alpha=0.001, max_iter=10000, tol=1e-1):
    x = x0
    path = [x]
    times = [0]
    start_time = time.time()
    for _ in range(max_iter):
        F = grad_U_pot(x,x_goal,obs,rho_0)
        x = x + alpha * F
        times.append(time.time() - start_time)
        path.append(x)
        if np.linalg.norm(x - x_goal) < tol:
            break
    final_time = time.time() - start_time
    return np.array(path),final_time,times


#路径
path_1 ,_,_= find_path(x0, x_goal,rho_01,obstacles)
path_2 ,_,_= find_path(x0, x_goal,rho_02,obstacles)
path_3 ,_,_= find_path(x0, x_goal,rho_03,obstacles)
path_4 ,_,_= find_path(x0, x_goal,rho_04,obstacles)
# 绘制
plt.figure(figsize=(10, 8))
plt.plot(x0[0], x0[1], 'ro', label='start')
plt.plot(x_goal[0], x_goal[1], 'go', label='end')
for obs in obstacles:
    circle = plt.Circle(obs['position'], obs['radius'], color='r', alpha=0.5, label='obstacles')
    plt.gca().add_artist(circle)
plt.plot(path_1[:, 0], path_1[:, 1], 'b-', label='rho0_1=1')
plt.plot(path_2[:, 0], path_2[:, 1], 'r-', label='rho0_2=0.5')
plt.plot(path_3[:, 0], path_3[:, 1], 'y-', label='rho0_3=0.25')
plt.plot(path_4[:, 0], path_4[:, 1], 'k-', label='rho0_4=0.1')
plt.title('path')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.show()
