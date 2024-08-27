import json
import os
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from Performance_Metrices import compute_metrics

def load_data(file_path):
    if file_path.endswith('.csv'):
        try:
            return pd.read_csv(file_path)
        except pd.errors.ParserError as e:
            print(f"Error parsing CSV file {file_path}: {e}")
            raise e
    elif file_path.endswith('.json'):
        try:
            with open(file_path, 'r') as file:
                return pd.DataFrame(json.load(file))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {e}")
            raise e
    else:
        raise ValueError("Unsupported file format")

def plot_individual_metrics_with_custom_legend(stats_dict, custom_labels):
    all_metrics = set()
    for stats in stats_dict.values():
        all_metrics.update(stats['mean'].index)
    print("All metrics:", all_metrics)
    all_metrics.discard('success')
    all_metrics.discard('success_rate')
    all_metrics.discard('collisions')
    all_metrics.discard('collision_rate')

    fig, axes = plt.subplots(len(all_metrics), 1, figsize=(10, 3.5 * len(all_metrics)))  # Adjusting size to make it more compact
    fig.suptitle('Performance Metrics', fontsize=16)

    bar_width = 0.35
    x = np.arange(len(stats_dict))

    if len(all_metrics) == 1:
        axes = [axes]

    def format_title(title):
        return title.replace('_', ' ').capitalize()

    for i, metric in enumerate(sorted(all_metrics)):
        ax = axes[i]
        for j, directory in enumerate(stats_dict.keys()):
            if metric in stats_dict[directory]['mean'].index:
                mean_val = stats_dict[directory]['mean'][metric]
                std_val = stats_dict[directory]['std'][metric]
                label = custom_labels.get(os.path.basename(directory), os.path.basename(directory))

                ax.bar(x[j], mean_val, bar_width, label=label)
                ax.errorbar(x[j], mean_val, yerr=std_val, fmt='o', color='black')

        ax.set_title(format_title(metric))
        ax.set_ylabel('Values')
        ax.set_xticks(x)
        ax.set_xticklabels([custom_labels.get(os.path.basename(dir), dir) for dir in stats_dict.keys()])

        # 设置相同的x轴范围
        ax.set_xlim(-0.5, len(stats_dict) - 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_rate(stats_dict, rate_type='success_rate', custom_labels=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    directories = list(stats_dict.keys())
    rates = [stats_dict[dir]['mean'][rate_type] for dir in directories]
    std_devs = [stats_dict[dir]['std'][rate_type] for dir in directories]

    bar_width = 0.4
    x = np.arange(len(directories))

    ax.bar(x, rates, bar_width, color='skyblue', yerr=std_devs, capsize=5)

    ax.set_ylabel(f'{rate_type.replace("_", " ").capitalize()}')
    ax.set_title(f'{rate_type.replace("_", " ").capitalize()}')

    if custom_labels is None:
        custom_labels = directories

    ax.set_xticks(x)
    ax.set_xticklabels([custom_labels.get(os.path.basename(dir), dir) for dir in directories])
    ax.set_ylim(0, 1.1)

    for i, v in enumerate(rates):
        ax.text(i, v + 0.03, f"{v:.2%}", ha='center', fontsize=12)

    plt.tight_layout()
    plt.show()

# Specify list of data directory paths
directories = [
    #'./APF_motion_planner_a_star',
    #'./APF_motion_planner_bfs',
    #'./APF_motion_planner_rrt',
    './CBF_motion_planner_a_star',
    './CBF_motion_planner_bfs',
    './CBF_motion_planner_rrt',
    #'./APF+CBF_motion_planner_a_star',
    #'./APF+CBF_motion_planner_bfs',
    #'./APF+CBF_motion_planner_rrt',
    #'./APF_csv',
    './CBF_csv',
    #'./CBF+APF_csv',
]

# Mapping of directories to custom labels
custom_labels = {
    'APF_motion_planner_a_star': 'APF a*',
    'APF_motion_planner_bfs': 'APF bfs',
    'APF_motion_planner_rrt': 'APF rrt',
    'CBF_motion_planner_a_star':'CBF a*',
    'CBF_motion_planner_bfs':'CBF bfs',
    'CBF_motion_planner_rrt':'CBF rrt',
    'APF_csv': 'APF',
    'CBF_csv': 'CBF',
    'CBF+APF_csv': 'APF+CBF',
    'APF_motion_planner_a_star': 'APF a*',
    'APF_motion_planner_bfs': 'APF bfs',
    'APF_motion_planner_rrt': 'APF rrt',
}

stats_dict = {}

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

        # Calculate mean and std for the metrics
        stats = df[['collisions', 'collision_rate', 'success', 'success_rate']].agg(['mean', 'std'])

        successful_df = df[df['success']]
        if 'human_fatigue' in successful_df.columns:
            successful_stats = successful_df[['task_completion_time', 'task_execution_time', 'smoothness', 'alignment', 'human_fatigue']].agg(['mean', 'std'])
        else:
            successful_stats = successful_df[['task_completion_time', 'task_execution_time', 'smoothness', 'alignment']].agg(['mean', 'std'])

        stats = pd.concat([successful_stats, stats], axis=1).T

        stats_dict[os.path.basename(directory_path)] = stats

# Plot performance metrics
plot_individual_metrics_with_custom_legend(stats_dict, custom_labels)

# Plot success rate with custom labels
plot_rate(stats_dict, rate_type='success_rate', custom_labels=custom_labels)

# Plot collision rate with custom labels
plot_rate(stats_dict, rate_type='collision_rate', custom_labels=custom_labels)
