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
    return data['time_steps'].max()

def compute_task_execution_times(data):
    return data['timestamp'].max()

def compute_human_fatigue(data):
    if 'key_pressed' in data.columns:
        key_presses = data['key_pressed'].notna().sum()
        return key_presses
    else:
        return None

def finite_differences(x, delta_t):
    velocities = []
    for i in range(1, len(x)):
        velocity = (x[i] - x[i - 1]) / delta_t[i]
        velocities.append(velocity)
    return np.array(velocities)

def compute_alignment(data):
    # Deviation from users intention
    positions = data['particle_position'].apply(lambda x: np.array(eval(x)))
    user_goals = data['user_goal'].apply(lambda x: np.array(eval(x)))

    delta_t = data['delta_t'].tolist()  # 计算时间步长

    particle_velocities = finite_differences(positions.tolist(), delta_t)
    user_goal_velocities = finite_differences(user_goals.tolist(), delta_t)

    min_length = min(len(particle_velocities), len(user_goal_velocities))
    particle_velocities = particle_velocities[:min_length]
    user_goal_velocities = user_goal_velocities[:min_length]

    velocity_differences = np.linalg.norm(particle_velocities - user_goal_velocities, axis=1)
    alignment = np.mean(velocity_differences)
    return alignment

def compute_smoothness(data):
    # Deviation from users intention
    velocity = data['velocity'].apply(lambda x: np.array(eval(x)))
    delta_t = data['delta_t']

    # Finit differences on velocity to get accelerations
    particle_acceleration = finite_differences(velocity.tolist(), delta_t.tolist())

    speed = np.linalg.norm(particle_acceleration, axis=1)
    smoothness = np.mean(speed)
    return smoothness

def compute_collisions(data):
    collisions = data['collision'].sum()
    return collisions

def compute_collision_rate(data):
    collisions = compute_collisions(data)
    total_samples = len(data)
    return collisions / total_samples if total_samples > 0 else 0

def compute_success(data):
    success = not any(data['success'] == False)
    return success

def compute_metrics(file_path):
    data = load_data(file_path)
    data['particle_position'] = data['particle_position'].apply(lambda x: str(x) if not isinstance(x, str) else x)

    task_completion_time = compute_task_completion_times(data)
    task_execution_time = compute_task_execution_times(data)
    human_fatigue = compute_human_fatigue(data)
    smoothness = compute_smoothness(data)
    alignment = compute_alignment(data)
    collisions = compute_collisions(data)
    collision_rate = compute_collision_rate(data)
    success = compute_success(data)

    metrics = {
        'task_completion_time': task_completion_time,
        'task_execution_time': task_execution_time,
        'smoothness': smoothness,
        'alignment': alignment,
        'collisions': collisions,
        'collision_rate': collision_rate,
        'success': success
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
        stats = df[['collisions', 'collision_rate', 'success']].agg(['mean', 'std', 'max', 'min'])

        # Filtering successful trajectories
        successful_df = df[df['success']]
        successful_stats = successful_df[['task_completion_time', 'task_execution_time', 'smoothness', 'alignment']].agg(['mean', 'std', 'max', 'min'])

        # Merging both statistics
        stats = pd.concat([successful_stats, stats], axis=1).T

        print("\nOverall Statistics:")
        for stat_name, stat_values in stats.items():
            print(f"\n{stat_name.upper()}:")
            print(stat_values)
    else:
        print("The provided path is neither a valid file nor a directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()
