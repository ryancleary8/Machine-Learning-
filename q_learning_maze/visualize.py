# ============================================================================
# FILE: visualize.py
# ============================================================================
"""
Visualization module for Q-Learning results.
Creates plots for maze layout, learning curves, Q-values, and agent paths.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_rewards(rewards_history, window=50, save_path='learning_curve.png'):
    """
    Plot learning curve showing average rewards over episodes.
    
    Args:
        rewards_history: List of total rewards per episode
        window: Moving average window size
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 5))
    
    # Calculate moving average
    rewards_smooth = np.convolve(rewards_history,
                                 np.ones(window)/window, mode='valid')
    
    plt.plot(rewards_smooth, linewidth=2, color='#2E86AB')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title(f'Learning Progress (Moving Average, window={window})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved comprehensive visualization to '{save_path}'")

