import json
import numpy as np
import pandas as pd
import sys

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            return pd.DataFrame(json.load(file))
    else:
        raise ValueError("不支持的文件格式")
def compute_task_completion_times(data):
    task_times = data['timestamp'].max() - data['timestamp'].min()
    return task_times

def compute_human_fatigue(data):
    key_presses = data['key_pressed'].notna().sum()
    return key_presses

def compute_smoothness(data):
    positions = data['particle_position'].apply(lambda x: np.array(eval(x)))
    diffs = np.diff(positions.tolist(), axis=0)
    smoothness = np.var(diffs)
    return smoothness

def compute_collisions(data):
    collisions = data['collision'].sum()
    return collisions


def main(file_path):
    data = load_data(file_path)

    # 确保 particle_position 列是字符串形式的列表，并转换为数组
    data['particle_position'] = data['particle_position'].apply(lambda x: str(x) if not isinstance(x, str) else x)

    # 计算各项指标
    task_completion_time = compute_task_completion_times(data)
    human_fatigue = compute_human_fatigue(data)
    smoothness = compute_smoothness(data)
    collisions = compute_collisions(data)

    # 合并结果到一个DataFrame中
    results = pd.DataFrame({
        'task_completion_time': [task_completion_time],
        'human_fatigue': [human_fatigue],
        'smoothness': [smoothness],
        'collisions': [collisions]
    })

    # 显示结果
    print(results)
    return results

# 示例用法
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    results = main(file_path)
