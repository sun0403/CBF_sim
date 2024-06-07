import matplotlib.pyplot as plt
import numpy as np
import APF,CBF,CBF_APF # 确保您的 APF 模块可以正常导入并使用

# 初始化目标位置和起始位置
x_goal = np.array([3.0, 5.0])
x_start = np.array([0.0, 0.0])

# APF 参数
rho_01 = 0.1
obstacles = [
    {'position': np.array([1.0, 2.0]), 'radius': 0.5},
    {'position': np.array([2.5, 3.0]), 'radius': 0.5}
]

# 初始化绘图
fig, ax = plt.subplots()
start_point, = ax.plot(x_start[0], x_start[1], 'ro', label='start')
goal_point, = ax.plot(x_goal[0], x_goal[1], 'ko', label='goal')
#path, final_time, times = APF.find_path(x_start, x_goal, rho_01,obstacles)#测试APF最简单情况
#需要进行寻路函数替代，复杂情况求解
#path, final_time, times=CBF.find_path_qp(x_start,x_goal,obstacles,alpha=0.5)
path,final_time,times=CBF_APF.find_path_v_star(x_start,x_goal,obstacles,rho_01,alpha=2)
path_line, = ax.plot(path[:, 0], path[:, 1], 'b-', label='path')

for obs in obstacles:
    circle = plt.Circle(obs['position'], obs['radius'], color='r', alpha=0.5)
    ax.add_artist(circle)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.legend()

# 更新绘图
def update_plot():
    global path_line, goal_point, x_goal
    print(f"Calculating path to new goal: {x_goal}")  # 打印目标点
    #path, final_time, times = APF.find_path(x_start, x_goal, rho_01,obstacles)
    #path, final_time, times = CBF.find_path_qp(x_start, x_goal, obstacles, alpha=2)
    #path,final_time,times=CBF_APF.find_path_v_star(x_start,x_goal,obstacles,rho_01,alpha=2)
    print(f"New path calculated: {path}")  # 打印新路径
    # 更新路径和目标点
    path_line.set_data(path[:, 0], path[:, 1])
    goal_point.set_data([x_goal[0]], [x_goal[1]])
    fig.canvas.draw_idle()

# 键盘事件处理
def on_key(event):
    global x_goal
    if event.key == 'up':
        x_goal[1] += 0.1
    elif event.key == 'down':
        x_goal[1] -= 0.1
    elif event.key == 'left':
        x_goal[0] -= 0.1
    elif event.key == 'right':
        x_goal[0] += 0.1
    update_plot()

# 使用交互模式
plt.ion()

# 绑定键盘事件
fig.canvas.mpl_connect('key_press_event', on_key)

# 显示初始绘图
update_plot()
plt.show(block=True)
