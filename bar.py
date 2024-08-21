import json
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from Performance_Metrices import compute_metrics

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

def plot_individual_metrics(stats_dict):
    # Collect all possible metrics across directories except 'success' and 'collision_rate'
    all_metrics = set()
    for stats in stats_dict.values():
        all_metrics.update(stats['mean'].index)
    print("All metrics:", all_metrics)
    all_metrics.discard('success')
    # Remove 'success' from the metrics to plot
    all_metrics.discard('success_rate')
    all_metrics.discard('collisions')  # Remove 'collisions' from the metrics to plot
    all_metrics.discard('collision_rate')  # Remove 'collision_rate' from the metrics to plot

    # Create subplots
    fig, axes = plt.subplots(len(all_metrics), 1, figsize=(14, 4 * len(all_metrics)))
    fig.suptitle('Performance Metrics', fontsize=16)

    bar_width = 0.15
    x = np.arange(3)  # max, min, mean

    # If there's only one metric, axes is not a list, make it a list for consistency
    if len(all_metrics) == 1:
        axes = [axes]

    # Plot each metric separately
    for i, metric in enumerate(sorted(all_metrics)):
        ax = axes[i]
        for j, directory in enumerate(stats_dict.keys()):
            if metric in stats_dict[directory]['mean'].index:
                max_val = stats_dict[directory]['max'][metric]
                min_val = stats_dict[directory]['min'][metric]
                mean_val = stats_dict[directory]['mean'][metric]
                std_val = stats_dict[directory]['std'][metric]
                values = [max_val, min_val, mean_val]
                ax.bar(x + j * bar_width, values, bar_width, label=directory)
                # Add error bar to mean value
                ax.errorbar(x[2] + j * bar_width, mean_val, yerr=std_val, fmt='o', color='black')

        ax.set_title(metric.capitalize())
        ax.set_ylabel('Values')
        ax.set_xticks(x + bar_width * (len(stats_dict) - 1) / 2)
        ax.set_xticklabels(['Max', 'Min', 'Mean'])
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def plot_success_rate(success_rates):
    # Create a bar plot for success rates
    fig, ax = plt.subplots(figsize=(10, 6))
    directories = list(success_rates.keys())
    success_values = [success_rates[dir] for dir in directories]

    bar_width = 0.35
    x = range(len(directories))

    ax.bar(x, success_values, bar_width)

    ax.set_xlabel('Directories')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate by Directory')
    ax.set_xticks(x)
    ax.set_xticklabels(directories, rotation=45, ha='right')
    ax.set_ylim(0, 1)

    for i, v in enumerate(success_values):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')

    plt.tight_layout()
    plt.show()

def plot_collision_rate(stats_dict):
    # Create a bar plot for collision rates
    fig, ax = plt.subplots(figsize=(10, 6))
    directories = list(stats_dict.keys())
    collision_rates = [stats_dict[dir]['mean']['collision_rate'] for dir in directories]

    bar_width = 0.35
    x = range(len(directories))

    ax.bar(x, collision_rates, bar_width)

    ax.set_xlabel('Directories')
    ax.set_ylabel('Collision Rate')
    ax.set_title('Collision Rate by Directory')
    ax.set_xticks(x)
    ax.set_xticklabels(directories, rotation=45, ha='right')
    ax.set_ylim(0, 1)

    for i, v in enumerate(collision_rates):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')

    plt.tight_layout()
    plt.show()

def plot_success_rate(stats_dict):
    # Create a bar plot for collision rates
    fig, ax = plt.subplots(figsize=(10, 6))
    directories = list(stats_dict.keys())
    success_rates = [stats_dict[dir]['mean']['success_rate'] for dir in directories]

    bar_width = 0.35
    x = range(len(directories))

    ax.bar(x, success_rates, bar_width)

    ax.set_xlabel('Directories')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(directories, rotation=45, ha='right')
    ax.set_ylim(0, 1)

    for i, v in enumerate(success_rates):
        ax.text(i, v + 0.02, f"{v:.2%}", ha='center')

    plt.tight_layout()
    plt.show()

# Specify list of data directory paths
directories = [
    './APF_csv',
    './CBF+APF_csv',
    './CBF_csv'
]

stats_dict = {}
success_rates = {}

for directory_path in directories:
    if not os.path.isdir(directory_path):
        print(f"The provided path {directory_path} is not a valid directory.")
        continue

    files = glob.glob(f"{directory_path}/*.csv")
    all_metrics = []



    for file in files:
        data = load_data(file)
        metrics = compute_metrics(file)
        all_metrics.append(metrics)
        print(f"Metrics for {file}: {metrics}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)

        # 首先计算已经存在的指标
        stats = df[['collisions', 'collision_rate', 'success','success_rate']].agg(['mean', 'std', 'max', 'min'])

        # 过滤成功的记录
        successful_df = df[df['success']]

        # 检查human_fatigue是否存在
        if 'human_fatigue' in successful_df.columns:
            # 如果存在，计算其统计信息
            successful_stats = successful_df[
                ['task_completion_time', 'task_execution_time', 'smoothness', 'alignment', 'human_fatigue']].agg(
                ['mean', 'std', 'max', 'min'])
        else:
            # 如果不存在，则不计算human_fatigue
            successful_stats = successful_df[
                ['task_completion_time', 'task_execution_time', 'smoothness', 'alignment']].agg(
                ['mean', 'std', 'max', 'min'])

        # 合并统计结果
        stats = pd.concat([successful_stats, stats], axis=1).T

        # 将统计信息保存到stats_dict中
        stats_dict[os.path.basename(directory_path)] = stats

# Plot performance metrics
plot_individual_metrics(stats_dict)


# Plot collision rate
plot_collision_rate(stats_dict)

plot_success_rate(stats_dict)