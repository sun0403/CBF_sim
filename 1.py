import matplotlib.pyplot as plt
import control as ctrl
import numpy as np

# 定义系统参数
omega_0 = 2 * np.pi * 5
D = 0.1

# 定义PI控制器的传递函数
PI = ctrl.TransferFunction([2, 1], [1, 0])

# 定义系统的开环传递函数
G_PT2 = ctrl.TransferFunction([1], [1, 2*D*omega_0, omega_0**2])

# 组合系统和控制器
open_loop = PI * G_PT2

# 绘制根轨迹图
fig, ax = plt.subplots()
ctrl.root_locus(open_loop, ax=ax)

# 设置标题在下方
plt.xlabel('Real Axis ')
plt.ylabel('Imaginary Axis ')
plt.grid(True)
plt.suptitle('')  # 清除默认的suptitle
fig.text(0.5, 0.02, r'Root Locus with a PI Controller (PI-Glied = $2 + \frac{1}{s}$)', ha='center')

plt.show()
