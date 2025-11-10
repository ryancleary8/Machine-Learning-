# ============================================================================
# FILE: config.py
# ============================================================================
"""
Configuration file for Q-Learning Grid-World Maze.
Stores all hyperparameters and environment settings in one place.
"""

import os

# Algorithm Selection
AGENT_ALGORITHM = "q_learning"  # Options: 'q_learning', 'sarsa', 'expected_sarsa',
                                #          'double_q', 'dqn', 'double_dqn'

# Directory for saving visualizations
RESULTS_DIR = os.path.join("results", AGENT_ALGORITHM)

# Environment Settings
GRID_SIZE = (13, 13)
START_POS = (0, 0)
GOAL_POS = (12, 12)

# Define obstacles (walls) as list of (row, col) tuples
OBSTACLES = [
    # Vertical walls
    (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
    # Horizontal walls
    (2, 1), (2, 2), (2, 3), (2, 4),
    (7, 5), (7, 6), (7, 7), (7, 8),
    # Additional obstacles
    (4, 5), (5, 5), (5, 1), (8,9), (7,9), (9,10)
]

# Reward Structure
REWARD_GOAL = 100      # Reward for reaching goal
REWARD_WALL = -10      # Penalty for hitting wall/boundary
REWARD_STEP = -1       # Small penalty for each step (encourages shorter paths)

# Q-Learning Hyperparameters
ALPHA = 0.1            # Learning rate (α)
GAMMA = 0.95           # Discount factor (γ)
EPSILON = 1.0          # Initial exploration rate (ε)
EPSILON_DECAY = 0.995  # Epsilon decay per episode
EPSILON_MIN = 0.01     # Minimum epsilon value

# Deep RL Hyperparameters (used by neural-network based agents)
DQN_LEARNING_RATE = 1e-3
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 5000
DQN_TARGET_UPDATE = 20  # Episodes between target network sync
DQN_HIDDEN_UNITS = (128, 128)

# Training Parameters
EPISODES = 1000        # Number of training episodes
MAX_STEPS = 200        # Maximum steps per episode

# Visualization Settings
PLOT_WINDOW = 50       # Moving average window for smoothing
SAVE_PATHS_AT = [0, 250, 500, 750, 999]  # Episodes to save paths for visualization
