import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import time
# 定义初始位置和目标位置define start position and goal
x0 = np.array([0.0, 0.0])
x_goal = np.array([3.0, 5.0])
K = 1
#定义CBF参数alpha和障碍 define parameter of CBF function and set alpha,obstacles
alpha_1 = 0.1
alpha_2 = 0.5
alpha_3 = 2
alpha_4 = 100
obstacles = [
    {'position': np.array([1.0, 2.0]), 'radius': 0.5},
    {'position': np.array([2.5, 3.0]), 'radius': 0.5}
]


##定义CBF define CBF function
def h(x, obstacles):
    h = []
    for obs in obstacles:
        h.append(np.linalg.norm(x - obs['position']) - 0.5)
    return h


#定义CBF梯度 define gradient of CBF
def grad_h(x, obstacles):
    grad_h = []
    for obs in obstacles:
        grad_h.append((x - obs['position']) / np.linalg.norm(x - obs['position']))
    return grad_h


#定义期望速度函数define function of desired volecity
def v_des(x, x_goal):
    return -K * (x - x_goal)


#定义QP问题并求解函数，用凸优化工具包cvxpy，也可以用ipopt
#define QP problem and solve it,by using convex python lib,can also use ipopt?
def qp_solver(x, x_goal, obstacles, alpha):
    v = cp.Variable(2)
    v_desired = v_des(x, x_goal)

    h_values = h(x, obstacles)
    grad_h_values = grad_h(x, obstacles)

    constraints = []
    for h_val, grad_h_val in zip(h_values, grad_h_values):
        constraints.append(grad_h_val @ v >= -alpha * h_val)

    objective = cp.Minimize(cp.sum_squares(v - v_desired))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS,verbose=True)

    return v.value


#定义寻路函数（APF类似）path finding function
def find_path_qp(x0, x_goal, obstacles, alpha, beta=0.01, max_iter=100000, tol=1e-3):
    x = x0
    path = [x]
    times = [0]
    start_time = time.time()
    for _ in range(max_iter):
        v_opt = qp_solver(x, x_goal, obstacles, alpha)
        x = x + beta * v_opt
        times.append(time.time() - start_time)
        path.append(x)
        if np.linalg.norm(x - x_goal) < tol:
            break
    final_time = time.time() - start_time
    return np.array(path),final_time,times


#路径规划计算 calculate path
path_1 ,_,_= find_path_qp(x0, x_goal, obstacles, alpha_1)
path_2,_,_ = find_path_qp(x0, x_goal, obstacles, alpha_2)
path_3 ,_,_= find_path_qp(x0, x_goal, obstacles, alpha_3)
path_4 ,_,_= find_path_qp(x0, x_goal, obstacles, alpha_4)

#绘制路径 show path in painting
plt.figure(figsize=(10, 8))
plt.plot(x0[0], x0[1], 'ro', label='start')
plt.plot(x_goal[0], x_goal[1], 'go', label='end')
for obs in obstacles:
    circle = plt.Circle(obs['position'], obs['radius'], color='r', alpha=0.5, label='obstacles')
    plt.gca().add_artist(circle)
plt.plot(path_1[:, 0], path_1[:, 1], 'b-', label='alpha_1=0.1')
plt.plot(path_2[:, 0], path_2[:, 1], 'r-', label='alpha_2=0.5')
plt.plot(path_3[:, 0], path_3[:, 1], 'y-', label='alpha_3=2')
plt.plot(path_4[:, 0], path_4[:, 1], 'k-', label='alpha_4=100')
plt.title('path')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.show()


