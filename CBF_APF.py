import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import time
import APF
#随机障碍
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
def rho(x, obs):
    return np.linalg.norm(x - obs['position']) - obs['radius']



#定义势函数及其梯度
def U_rep(x,obtacles,rho_0):
    for obs in obstacles:
        if rho(x,obs)<rho_0:
            U_obs=0.5*K_rep*(1/rho(x,obs)-1/rho_0)**2
        else:
            U_obs=0
    return U_obs
def grad_U_rep(x,obstacles,rho_0):
    grad_U_obs = np.zeros_like(x, dtype=np.float64)
    for obs in obstacles:
        rho_x = rho(x, obs)
        if rho_x <= rho_0:
            grad_U_obs += K_rep * (1/rho_x - 1/rho_0) * (-1/rho_x**2) * (x - obs['position']) / np.linalg.norm(x - obs['position'])
        else:
            grad_U_obs += 0

    return grad_U_obs
def U_attra(x,x_goal):
    return

def grad_U_att(x, x_goal):
    return K_att * (x - x_goal)


#定义CBF和gradient
def h(x, obstacles, delta,rho_0):
    U_rep_value = U_rep(x, obstacles,rho_0)
    return 1 / (1 + U_rep_value) - delta
def grad_h(x, obstacles,rho_0):
    U_rep_value = U_rep(x, obstacles,rho_0)
    grad_U_rep_value = grad_U_rep(x, obstacles,rho_0)
    return -grad_U_rep_value / (1 + U_rep_value)**2

#利用论文给的显式解
def v_star(x, x_goal, obstacles, alpha, delta,rho_0):
    v_att = -grad_U_att(x, x_goal)
    h_value = h(x, obstacles, delta,rho_0)
    grad_h_value = grad_h(x, obstacles,rho_0)
    norm_grad_h_value = np.dot(grad_h_value, grad_h_value)

    if norm_grad_h_value == 0:
        # 如果梯度为零，直接返回吸引力
        return v_att

    Phi = np.dot(grad_h_value, v_att) + alpha * h_value

    if Phi < 0:
        v = v_att + (-Phi / np.dot(grad_h_value, grad_h_value)) * grad_h_value
    else:
        v = v_att

    return v
#find path
def find_path_v_star(x0, x_goal, obstacles,rho_0,alpha, delta=0.001, beta=0.01, max_iter=10000, tol=1e-2):
    x = x0
    path = [x]
    times = [0]
    start_time = time.time()
    for _ in range(max_iter):
        v_opt = v_star(x, x_goal, obstacles, alpha, delta,rho_0)
        x = x + beta * v_opt
        times.append(time.time() - start_time)
        path.append(x)
        if np.linalg.norm(x - x_goal) < tol:
            break

    final_time = time.time() - start_time

    return np.array(path),final_time,times

# 定义初始位置和目标位置
x0 = np.array([0.0, 0.0])
x_goal = np.array([3.0, 5.0])

alpha=1
K_att=1
K_rep=1
rho_01=0.5
rho_02=0.1
delta=0.001
num_obstacles=5

'''obstacles = [
    {'position': np.array([1.0, 2.0]), 'radius': 0.5},
    {'position': np.array([2.5, 3.0]), 'radius': 0.5}]'''
obstacles=generate_random_obstacles(num_obstacles)
path_rho_01,_,_ = find_path_v_star(x0, x_goal, obstacles,rho_01, alpha, delta)
path_rho_02,_,_ = find_path_v_star(x0, x_goal, obstacles,rho_02,alpha, delta)
##path_1,_,_=APF.find_path(x0,x_goal,rho_01,obstacles)
##path_4,_,_=APF.find_path(x0,x_goal,rho_02,obstacles)


plt.figure(figsize=(10, 8))
plt.plot(x0[0], x0[1], 'ro', label='start')
plt.plot(x_goal[0], x_goal[1], 'go', label='end')
for obs in obstacles:
    circle = plt.Circle(obs['position'], obs['radius'], color='r', alpha=0.5, label='obstacle')
    plt.gca().add_artist(circle)
plt.plot(path_rho_01[:, 0], path_rho_01[:, 1], 'b-', label='CBF-APF rho0=1')
plt.plot(path_rho_02[:, 0], path_rho_02[:, 1], 'r-', label='CBF-APF rho0=0.1')
##plt.plot(path_1[:, 0], path_1[:, 1], 'y-', label='APF rho0=1')
##plt.plot(path_4[:, 0], path_4[:, 1], 'k-', label='APF rho0=0.1')
plt.title('Path Planning with CBF based APF')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.show()
