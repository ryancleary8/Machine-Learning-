# ============================================================================
# FILE: visualize.py
# ============================================================================
"""
Visualization helpers for the Q-learning maze project.

Creates plots for maze layout, learning curves, Q-values, and agent paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def _moving_average(values, window):
    """Safely compute a moving average for short sequences."""
    if len(values) == 0:
        return np.array([])

    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode='valid')


def plot_rewards(rewards_history, window=50, save_path='learning_curve.png'):
    """Plot the moving-average reward obtained per episode."""
    plt.figure(figsize=(10, 5))

    rewards_smooth = _moving_average(rewards_history, window)
    episodes = np.arange(len(rewards_smooth)) + 1
    actual_window = max(1, min(window, len(rewards_history))) if len(rewards_history) > 0 else 1

    if rewards_smooth.size > 0:
        plt.plot(episodes, rewards_smooth, linewidth=2, color='#2E86AB')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(f'Learning Progress (Moving Average, window={actual_window})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved reward curve to '{save_path}'")


def plot_steps(steps_history, window=50, save_path='steps_curve.png'):
    """Plot the moving-average number of steps taken per episode."""
    plt.figure(figsize=(10, 5))

    steps_smooth = _moving_average(steps_history, window)
    episodes = np.arange(len(steps_smooth)) + 1
    actual_window = max(1, min(window, len(steps_history))) if len(steps_history) > 0 else 1

    if steps_smooth.size > 0:
        plt.plot(episodes, steps_smooth, linewidth=2, color='#FF6F59')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Steps', fontsize=12)
    plt.title(f'Steps per Episode (Moving Average, window={actual_window})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved steps curve to '{save_path}'")


def show_q_heatmap(q_table, env, save_path='q_heatmap.png'):
    """Create a heatmap of max-Q values across the grid."""
    max_q_values = np.max(q_table, axis=1).reshape(env.rows, env.cols)

    # Mask obstacles for clearer visualization
    obstacle_mask = env.grid == 1
    max_q_values = np.ma.array(max_q_values, mask=obstacle_mask)

    plt.figure(figsize=(8, 6))
    cmap = plt.cm.viridis
    im = plt.imshow(max_q_values, cmap=cmap, origin='upper')
    im.cmap.set_bad('black')
    plt.colorbar(im, label='Max Q-Value')
    plt.title('Max Q-Values Heatmap')
    plt.xticks(range(env.cols))
    plt.yticks(range(env.rows))

    # Annotate start and goal
    plt.text(env.start[1], env.start[0], 'S', ha='center', va='center',
             color='white', fontsize=12, fontweight='bold')
    plt.text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
             color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Q-value heatmap to '{save_path}'")


def plot_policy(q_table, env, save_path='policy.png'):
    """Plot the greedy policy derived from the Q-table."""
    best_actions = np.argmax(q_table, axis=1).reshape(env.rows, env.cols)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.set_xticklabels(range(env.cols))
    ax.set_yticklabels(range(env.rows))
    ax.grid(True, which='both', color='lightgray', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title('Greedy Policy (Best Actions)')

    arrow_map = {
        0: (0, -0.3),   # Up
        1: (0, 0.3),    # Down
        2: (-0.3, 0),   # Left
        3: (0.3, 0),    # Right
    }

    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row, col] == 1:
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='black', alpha=0.6)
                ax.add_patch(rect)
                continue
            if (row, col) == env.goal:
                ax.text(col, row, 'G', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='green')
                continue
            if (row, col) == env.start:
                ax.text(col, row, 'S', ha='center', va='center',
                        fontsize=12, fontweight='bold', color='blue')

            action = best_actions[row, col]
            dx, dy = arrow_map[action]
            ax.arrow(col, row, dx, dy, head_width=0.15, head_length=0.15,
                     fc='red', ec='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved greedy policy plot to '{save_path}'")


def plot_maze_with_path(env, path, title='Agent Path', save_path='maze_path.png'):
    """Plot the maze layout and overlay the agent's path."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Draw grid background
    ax.set_xlim(-0.5, env.cols - 0.5)
    ax.set_ylim(-0.5, env.rows - 0.5)
    ax.set_xticks(range(env.cols))
    ax.set_yticks(range(env.rows))
    ax.grid(True, which='both', color='lightgray', linewidth=0.5)
    ax.invert_yaxis()
    ax.set_title(title)

    # Draw obstacles
    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row, col] == 1:
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='black', alpha=0.7)
                ax.add_patch(rect)

    # Mark start and goal
    ax.text(env.start[1], env.start[0], 'S', ha='center', va='center',
            fontsize=12, fontweight='bold', color='blue')
    ax.text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
            fontsize=12, fontweight='bold', color='green')

    if path is not None and len(path) > 0:
        path_rows = [state[0] for state in path]
        path_cols = [state[1] for state in path]
        ax.plot(path_cols, path_rows, color='#FFA500', linewidth=2,
                marker='o', markersize=4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved maze path plot to '{save_path}'")


def create_comprehensive_visualization(env, agent, rewards_history,
                                       steps_history, paths_history,
                                       save_path='comprehensive_results.png'):
    """Create a single figure summarising key training artefacts."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Learning curve
    rewards_smooth = _moving_average(rewards_history, window=50)
    if rewards_smooth.size > 0:
        episodes = np.arange(len(rewards_smooth)) + 1
        axes[0, 0].plot(episodes, rewards_smooth, color='#2E86AB')
    axes[0, 0].set_title('Average Reward (Moving Average)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)

    # Steps curve
    steps_smooth = _moving_average(steps_history, window=50)
    if steps_smooth.size > 0:
        episodes_steps = np.arange(len(steps_smooth)) + 1
        axes[0, 1].plot(episodes_steps, steps_smooth, color='#FF6F59')
    axes[0, 1].set_title('Steps per Episode (Moving Average)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].grid(True, alpha=0.3)

    # Max Q heatmap
    max_q_values = np.max(agent.q_table, axis=1).reshape(env.rows, env.cols)
    obstacle_mask = env.grid == 1
    max_q_values = np.ma.array(max_q_values, mask=obstacle_mask)
    cmap = plt.cm.viridis
    im = axes[0, 2].imshow(max_q_values, cmap=cmap, origin='upper')
    im.cmap.set_bad('black')
    axes[0, 2].set_title('Max Q-Values Heatmap')
    axes[0, 2].set_xticks(range(env.cols))
    axes[0, 2].set_yticks(range(env.rows))
    axes[0, 2].set_xlabel('Column')
    axes[0, 2].set_ylabel('Row')
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Policy arrows
    best_actions = np.argmax(agent.q_table, axis=1).reshape(env.rows, env.cols)
    axes[1, 0].set_xlim(-0.5, env.cols - 0.5)
    axes[1, 0].set_ylim(-0.5, env.rows - 0.5)
    axes[1, 0].set_xticks(range(env.cols))
    axes[1, 0].set_yticks(range(env.rows))
    axes[1, 0].grid(True, which='both', color='lightgray', linewidth=0.5)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_title('Greedy Policy')
    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row, col] == 1:
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='black', alpha=0.6)
                axes[1, 0].add_patch(rect)
                continue
            if (row, col) == env.goal:
                axes[1, 0].text(col, row, 'G', ha='center', va='center',
                                fontsize=12, fontweight='bold', color='green')
                continue
            if (row, col) == env.start:
                axes[1, 0].text(col, row, 'S', ha='center', va='center',
                                fontsize=12, fontweight='bold', color='blue')

            action = best_actions[row, col]
            dx, dy = {
                0: (0, -0.3),
                1: (0, 0.3),
                2: (-0.3, 0),
                3: (0.3, 0),
            }[action]
            axes[1, 0].arrow(col, row, dx, dy, head_width=0.15,
                             head_length=0.15, fc='red', ec='red')

    # Maze with final saved path (if available)
    axes[1, 1].set_xlim(-0.5, env.cols - 0.5)
    axes[1, 1].set_ylim(-0.5, env.rows - 0.5)
    axes[1, 1].set_xticks(range(env.cols))
    axes[1, 1].set_yticks(range(env.rows))
    axes[1, 1].grid(True, which='both', color='lightgray', linewidth=0.5)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title('Sample Episode Path')

    for row in range(env.rows):
        for col in range(env.cols):
            if env.grid[row, col] == 1:
                rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                         facecolor='black', alpha=0.7)
                axes[1, 1].add_patch(rect)

    axes[1, 1].text(env.start[1], env.start[0], 'S', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='blue')
    axes[1, 1].text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
                    fontsize=12, fontweight='bold', color='green')

    sample_path = None
    if isinstance(paths_history, dict) and len(paths_history) > 0:
        last_episode = sorted(paths_history.keys())[-1]
        sample_path = paths_history[last_episode]
    elif isinstance(paths_history, (list, tuple)) and len(paths_history) > 0:
        sample_path = paths_history[-1]

    if sample_path is not None and len(sample_path) > 0:
        path_rows = [state[0] for state in sample_path]
        path_cols = [state[1] for state in sample_path]
        axes[1, 1].plot(path_cols, path_rows, color='#FFA500', linewidth=2,
                        marker='o', markersize=4)

    # Remove unused subplot (bottom-right)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive visualization to '{save_path}'")

