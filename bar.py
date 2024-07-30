import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from Performance_Metrices import compute_metrics

def plot_individual_metrics(stats_dict):
    # Collect all possible metrics across directories
    all_metrics = set()
    for stats in stats_dict.values():
        all_metrics.update(stats['mean'].index)

    # Create subplots
    fig, axes = plt.subplots(len(all_metrics), 1, figsize=(14, 4 * len(all_metrics)))
    fig.suptitle('Performance Metrics', fontsize=16)

    bar_width = 0.15
    x = np.arange(3)  # max, min, mean

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

# 指定数据目录路径列表
directories = [
    '/home/sun/CBF_sim/APF+CBF_motion_planner',
    '/home/sun/CBF_sim/CBF_motion_planner',
'/home/sun/CBF_sim/APF_motion_planner',
]

stats_dict = {}

for directory_path in directories:
    if not os.path.isdir(directory_path):
        print(f"The provided path {directory_path} is not a valid directory.")
        continue

    files = glob.glob(f"{directory_path}/*.csv")
    all_metrics = []

    for file in files:
        metrics = compute_metrics(file)
        all_metrics.append(metrics)
        print(f"Metrics for {file}: {metrics}")

    if all_metrics:
        df = pd.DataFrame(all_metrics)

        stats = {
            'mean': df.mean(),
            'min': df.min(),
            'max': df.max(),
            'std': df.std()
        }

        stats_dict[os.path.basename(directory_path)] = stats
    else:
        print(f"No CSV files found in directory {directory_path}.")

# 排除没有人类疲劳度数据的文件
filtered_stats_dict = {
    key: value for key, value in stats_dict.items()
    if 'human_fatigue' not in value['mean'].index or key not in ['A_star_path', 'bfs_path', 'rrt_path']
}

plot_individual_metrics(filtered_stats_dict)
