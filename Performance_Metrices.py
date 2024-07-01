import json
import numpy as np
import pandas as pd
import sys
import glob
import os

def load_data(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as file:
            return pd.DataFrame(json.load(file))
    else:
        raise ValueError("Unsupported file format")

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

def compute_metrics(file_path):
    data = load_data(file_path)
    data['particle_position'] = data['particle_position'].apply(lambda x: str(x) if not isinstance(x, str) else x)

    task_completion_time = compute_task_completion_times(data)
    human_fatigue = compute_human_fatigue(data)
    smoothness = compute_smoothness(data)
    collisions = compute_collisions(data)

    return {
        'task_completion_time': task_completion_time,
        'human_fatigue': human_fatigue,
        'smoothness': smoothness,
        'collisions': collisions
    }

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
