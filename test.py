import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# 从 bezier_path.py 复制相关代码
def calc_4points_bezier_path(sx, sy, syaw, gx, gy, gyaw, offset):
    dist = np.hypot(sx - gx, sy - gy) / offset
    control_points = np.array(
        [[sx, sy],
         [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],
         [gx - dist * np.cos(gyaw), gy - dist * np.sin(gyaw)],
         [gx, gy]])
    path = calc_bezier_path(control_points, n_points=100)
    return path, control_points

def calc_bezier_path(control_points, n_points=100):
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))
    return np.array(traj)

def Comb(n, i, t):
    return comb(n, i) * t ** i * (1 - t) ** (n - i)

def bezier(t, control_points):
    n = len(control_points) - 1
    return np.sum([Comb(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)

def draw_arrow(x, y, yaw, length=1.0, width=0.5, color='r'):
    plt.arrow(x, y, length * np.cos(yaw), length * np.sin(yaw),
              head_length=width, head_width=width, color=color)

def main():
    # 设置起点和终点
    sx, sy, syaw = 10.0, 1.0, np.deg2rad(180.0)
    gx, gy, gyaw = 0.0, -3.0, np.deg2rad(-45.0)
    offset = 3.0

    # 计算Bézier路径和控制点
    path, control_points = calc_4points_bezier_path(sx, sy, syaw, gx, gy, gyaw, offset)

    # 绘制路径和控制点
    plt.plot(path[:, 0], path[:, 1], label="Bézier Path")
    plt.plot(control_points[:, 0], control_points[:, 1], '--o', label="Control Points")
    draw_arrow(sx, sy, syaw)
    draw_arrow(gx, gy, gyaw)
    plt.grid(True)
    plt.axis("equal")
    plt.title("Bézier Path Example")
    plt.legend()
    plt.show()

# 运行主函数
if __name__ == '__main__':
    main()

