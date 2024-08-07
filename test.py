import os
import pandas as pd

# 定义数据
data = {
    "timestamp": [0, 1, 2],
    "particle_position": [[0, 0], [1, 1], [2, 2]],
    "user_goal": [[1, 1], [2, 2], [3, 3]],
    "velocity": [[0, 0], [0.5, 0.5], [1, 1]],
    "collision": [False, False, True],
    "success": [True, True, False]
}

df = pd.DataFrame(data)

# 定义相对路径
relative_path = f"./CBF_sim/APF+CBF_motion_planner/00.csv"

# 打印当前工作目录
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")

# 打印相对路径进行调试
print(f"Relative path: {relative_path}")

# 获取绝对路径
absolute_path = os.path.abspath(relative_path)
print(f"Absolute path: {absolute_path}")

# 确保目录存在
os.makedirs(os.path.dirname(absolute_path), exist_ok=True)

# 保存文件
df.to_csv(absolute_path, index=False)
print(f"Data saved to {absolute_path}")
