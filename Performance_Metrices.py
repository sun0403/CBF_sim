import numpy as np
import time
import CBF_APF
import APF

# 参数设置
alpha1 = 0.5
num_obstacles = 3
num_do = 100
rho_0 = 0.5

# 起点和终点位置
x0 = np.array([0.0, 0.0])
x_goal = np.array([4.0, 5.0])


# 检查点是否与障碍物碰撞
def check_collision(point, obstacles):
    for obs in obstacles:
        if np.linalg.norm(point - obs['position']) <= obs['radius']:
            return True
    return False


# 计算路径的光滑度
def calculate_smoothness(path):
    # 计算速度
    velocities = np.diff(path, axis=0)
    # 计算加速度
    accelerations = np.diff(velocities, axis=0)
    # 计算加加速度（Jerk）
    #jerks = np.diff(accelerations, axis=0)
    # 返回加加速度的平均值作为光滑度的度量，值越小越光滑
    return np.mean(np.linalg.norm(accelerations, axis=1))


# 评估路径规划算法性能
def evaluate_performance(num_obstacles, num_trials, x0, x_goal, alpha):
    position_errors = []
    completion_times = []
    computation_times = []
    smoothness_values = []
    collisions = 0

    for _ in range(num_trials):
        # 生成随机障碍物
        obstacles = CBF_APF.generate_random_obstacles(num_obstacles)

        # 路径规划
        #path, final_time, times = CBF_APF.find_path_v_star(x0, x_goal, obstacles, rho_0, alpha1, delta=0.001, beta=0.01, max_iter=100000, tol=1e-2)  # CBF_APF
        # path, final_time, times = CBF.find_path_qp(x0, x_goal, obstacles, alpha1, beta=0.01, max_iter=100000, tol=1e-3)  # CBF
        path, final_time, times = APF.find_path(x0, x_goal, rho_0, obstacles, alpha=alpha1, max_iter=10000,tol=1e-2)  # APF

        # 计算位置误差
        position_error = np.linalg.norm(path[-1] - x_goal)
        position_errors.append(position_error)

        # 记录完成时间
        completion_times.append(final_time)

        # 计算平均计算时间
        avg_computation_time = np.mean(np.diff(times))
        computation_times.append(avg_computation_time)

        # 检查路径中的碰撞次数
        for point in path:
            if check_collision(point, obstacles):
                collisions += 1
                break

        # 计算路径的光滑度
        smoothness = calculate_smoothness(path)
        smoothness_values.append(smoothness)

    # 计算平均性能指标
    metrics = {
        "Position Error": np.mean(position_errors),
        "Completion Time": np.mean(completion_times),
        "Average Computation Time": np.mean(computation_times),
        "Number of Collisions": collisions,
        "Smoothness (Jerk)": np.mean(smoothness_values)
    }

    return metrics


# 评估算法性能
metrics = evaluate_performance(num_obstacles, num_do, x0, x_goal, alpha1)
print(metrics)
