import matplotlib.pyplot as plt
import numpy as np

# Metrics dictionary
stats_dict = {
    'APF': {
        'task_completion_time': 0.0017127990722656,
        'task_execution_time': 14.958674907684326,
        'smoothness': 1142.425896629985,
        'alignment': 91.68546905404799,
        'collisions': 0,
        'collision_rate': 0.0,
        'success': True,
        'success_rate': 1.0,
        'human_fatigue': 264
    },
    'CBF': {
        'task_completion_time': 0.0093717575073242,
        'task_execution_time': 10.885072946548462,
        'smoothness': 676.5566562918513,
        'alignment': 13.569569292790934,
        'collisions': 1,
        'collision_rate': 0.00546448087431694,
        'success': True,
        'success_rate': 1.0,
        'human_fatigue': 175
    },
    'APF+CBF': {
        'task_completion_time': 0.0010397434234619,
        'task_execution_time': 10.413661241531372,
        'smoothness': 940.3033229183113,
        'alignment': 15.636470260463245,
        'collisions': 0,
        'collision_rate': 0.0,
        'success': True,
        'success_rate': 1.0,
        'human_fatigue': 160
    },
}

# Define colors for each method based on the new image
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

# Plot individual metrics
def plot_individual_metrics(stats_dict):
    all_metrics = set()
    for stats in stats_dict.values():
        all_metrics.update(stats.keys())

    # Discard non-relevant metrics for plotting
    all_metrics.discard('success')
    all_metrics.discard('success_rate')
    all_metrics.discard('collisions')
    all_metrics.discard('collision_rate')

    fig, axes = plt.subplots(len(all_metrics), 1, figsize=(14, 4 * len(all_metrics)))
    fig.suptitle('Performance Metrics Comparison', fontsize=16)

    bar_width = 0.25
    x = np.arange(len(stats_dict))

    if len(all_metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(sorted(all_metrics)):
        ax = axes[i]
        values = [stats_dict[method].get(metric, np.nan) for method in stats_dict.keys()]
        bars = ax.bar(x, values, bar_width, color=colors)

        # 修改标题，去掉下划线并首字母大写
        formatted_metric = metric.replace('_', ' ').capitalize()
        ax.set_title(formatted_metric)

        ax.set_ylabel('Values')
        ax.set_xticks(x)
        ax.set_xticklabels(list(stats_dict.keys()), rotation=45, ha='right')
        ax.set_ylim(0, max(values) * 1.1)  # Set y-axis limit slightly higher than max value

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # 自定义图例的位置
    fig.legend(bars, stats_dict.keys(), loc='center left', bbox_to_anchor=(0.89, 0.9), fontsize=12)
    plt.show()


# Plot rate metrics
def plot_rate(stats_dict, rate_type='success_rate', new_labels=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    files = list(stats_dict.keys())
    rates = [stats_dict[file].get(rate_type, np.nan) for file in files]

    bar_width = 0.35
    x = np.arange(len(files))  # Ensure this matches the number of files

    bars = ax.bar(x, rates, bar_width, color=colors)

    # 修改标题，去掉下划线并首字母大写
    formatted_rate_type = rate_type.replace('_', ' ').capitalize()
    ax.set_ylabel(formatted_rate_type)
    ax.set_title(formatted_rate_type)

    if new_labels is None or len(new_labels) != len(files):
        new_labels = files  # Fallback to using file names if lengths don't match

    ax.set_xticks(x)
    ax.set_xticklabels(new_labels, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)

    for i, v in enumerate(rates):
        ax.text(i, v + 0.03, f"{v:.2%}", ha='center', fontsize=12)

    # 自定义图例的位置
    ax.legend(bars, stats_dict.keys(), loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    plt.tight_layout()
    plt.show()


# Custom labels for the X-axis (optional)
new_labels = ['APF', 'CBF', 'CBF+APF']

# Plot performance metrics
plot_individual_metrics(stats_dict)

# Plot success rate with custom labels
plot_rate(stats_dict, rate_type='success_rate', new_labels=new_labels)

# Plot collision rate with custom labels
plot_rate(stats_dict, rate_type='collision_rate', new_labels=new_labels)
