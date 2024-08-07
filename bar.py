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
    # Collect all possible metrics across directories except 'success'
    all_metrics = set()
    for stats in stats_dict.values():
        all_metrics.update(stats['mean'].index)
    all_metrics.discard('success')  # Remove 'success' from the metrics to plot

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

# Specify list of data directory paths
directories = [
    '/Users/yuanzhengsun/Desktop/CBF_sim/CBF/APF_motion_planner',
    '/Users/yuanzhengsun/Desktop/CBF_sim/CBF/CBF_motion_planner',
    '/Users/yuanzhengsun/Desktop/CBF_sim/CBF/APF+CBF_motion_planner',

]

stats_dict = {}
success_rates = {}

for directory_path in directories:
    if not os.path.isdir(directory_path):
        print(f"The provided path {directory_path} is not a valid directory.")
        continue

    files = glob.glob(f"{directory_path}/*.csv")
    all_metrics = []

    success_count = 0
    total_count = 0

    for file in files:
        data = load_data(file)

        # Ensure 'success' column exists
        if 'success' not in data.columns:
            print(f"File {file} does not contain 'success' column. Skipping...")
            continue

        metrics = compute_metrics(file)
        all_metrics.append(metrics)
        print(f"Metrics for {file}: {metrics}")

        if metrics['success']:
            success_count += 1
        total_count += 1

    if all_metrics:
        df = pd.DataFrame(all_metrics)

        stats = {
            'mean': df.mean(),
            'min': df.min(),
            'max': df.max(),
            'std': df.std()
        }

        stats_dict[os.path.basename(directory_path)] = stats
        success_rates[os.path.basename(directory_path)] = success_count / total_count if total_count > 0 else 0
    else:
        print(f"No CSV files found in directory {directory_path}.")

# Plot performance metrics
plot_individual_metrics(stats_dict)

# Plot success rate
plot_success_rate(success_rates)
