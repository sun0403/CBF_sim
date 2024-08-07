import json
import numpy as np
import pandas as pd
import sys
import glob
import os

def load_data(file_path):
    if file_path.endswith('.csv'):
        try:
            return pd.read_csv(file_path)
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file {file_path}: {e}")
            with open(file_path, 'r') as file:
                for i, line in enumerate(file, 1):
                    try:
                        pd.read_csv(pd.compat.StringIO(line))
                    except pd.errors.ParserError:
                        print(f"Error parsing line {i} in file {file_path}")
                        break
            raise e
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            return pd.DataFrame(json.load(file))
    else:
        raise ValueError("Unsupported file format")

def compute_task_completion_times(data):
    task_times = data['timestamp'].max() - data['timestamp'].min()
    return task_times

def compute_human_fatigue(data):
    if 'key_pressed' in data.columns:
        key_presses = data['key_pressed'].notna().sum()
        return key_presses
    else:
        return None
def compute_velocity(positions, delta_t):
    velocities = []
    for i in range(1, len(positions)):
        velocity = (positions[i] - positions[i - 1]) / delta_t
        velocities.append(velocity)
    return np.array(velocities)


def compute_smoothness_old(data):
    # Deviation from users intention
    positions = data['particle_position'].apply(lambda x: np.array(eval(x)))
    user_goals = data['user_goal'].apply(lambda x: np.array(eval(x)))

    delta_t = data['timestamp'].diff().mean()  # 计算时间步长

    particle_velocities = compute_velocity(positions.tolist(), delta_t)
    user_goal_velocities = compute_velocity(user_goals.tolist(), delta_t)


    min_length = min(len(particle_velocities), len(user_goal_velocities))
    particle_velocities = particle_velocities[:min_length]
    user_goal_velocities = user_goal_velocities[:min_length]

    velocity_differences = np.linalg.norm(particle_velocities - user_goal_velocities, axis=1)
    smoothness = np.sum(velocity_differences)
    return smoothness

def compute_smoothness(data):
    # Deviation from users intention
    velocity = data['velocity'].apply(lambda x: np.array(eval(x)))

    delta_t = data['timestamp'].diff().mean()  # 计算时间步长

    # Finit differences on velocity to get accelerations
    particle_acceleration = compute_velocity(velocity.tolist(), delta_t)
    speed = np.linalg.norm(particle_acceleration, axis=1)
    smoothness = np.sum(speed)
    return smoothness

def compute_collisions(data):
    collisions = data['collision'].sum()
    return collisions
def compute_success(data):
    success = not any(data['success'] == False)
    return success

def compute_metrics(file_path):
    data = load_data(file_path)
    data['particle_position'] = data['particle_position'].apply(lambda x: str(x) if not isinstance(x, str) else x)

    task_completion_time = compute_task_completion_times(data)
    human_fatigue = compute_human_fatigue(data)
    smoothness = compute_smoothness(data)
    collisions = compute_collisions(data)
    success=compute_success(data)

    metrics = {
        'task_completion_time': task_completion_time,
        'smoothness': smoothness,
        'collisions': collisions,
        'success':success
    }

    if human_fatigue is not None:
        metrics['human_fatigue'] = human_fatigue

    return metrics

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path_or_directory>")
        sys.exit(1)

    path = sys.argv[1]

    if os.path.isfile(path):
        metrics = compute_metrics(path)
        print(f"Metrics for {path}: {metrics}")
    elif os.path.isdir(path):
        files = glob.glob(f"{path}/*.csv")
        all_metrics = []

        for file in files:
            metrics = compute_metrics(file)
            all_metrics.append(metrics)
            print(f"Metrics for {file}: {metrics}")

        df = pd.DataFrame(all_metrics)

        stats = {
            'min': df.min(),
            'max': df.max(),
            'mean': df.mean(),
            'std': df.std()
        }

        print("\nOverall Statistics:")
        for stat_name, stat_values in stats.items():
            print(f"\n{stat_name.upper()}:")
            print(stat_values)
    else:
        print("The provided path is neither a valid file nor a directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
