"""
Q-Learning Grid-World Maze Solver - Modular Implementation
===========================================================

Project Structure:
q_learning_maze/
├── main.py
├── environment.py
├── agent_qlearning.py
├── agent_base.py
├── visualize.py
└── config.py
"""

# ============================================================================
# FILE: config.py
# ============================================================================
"""
Configuration file for Q-Learning Grid-World Maze.
Stores all hyperparameters and environment settings in one place.
"""

# Environment Settings
GRID_SIZE = (10, 10)
START_POS = (0, 0)
GOAL_POS = (9, 9)

# Define obstacles (walls) as list of (row, col) tuples
OBSTACLES = [
    # Vertical walls
    (2, 3), (3, 3), (4, 3), (5, 3), (6, 3), (7, 3),
    (2, 7), (3, 7), (4, 7), (5, 7), (6, 7), (7, 7),
    # Horizontal walls
    (2, 1), (2, 2), (2, 3), (2, 4),
    (7, 5), (7, 6), (7, 7), (7, 8),
    # Additional obstacles
    (4, 5), (5, 5), (5, 1)
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

# Training Parameters
EPISODES = 1000        # Number of training episodes
MAX_STEPS = 200        # Maximum steps per episode

# Visualization Settings
PLOT_WINDOW = 50       # Moving average window for smoothing
SAVE_PATHS_AT = [0, 250, 500, 750, 999]  # Episodes to save paths for visualization


# ============================================================================
# FILE: agent_base.py
# ============================================================================
"""
Abstract base class for reinforcement learning agents.
Defines the common interface that all agents must implement.
"""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Abstract base class for RL agents.
    
    All agents must implement choose_action() and update() methods.
    This ensures a consistent interface across different RL algorithms.
    """
    
    def __init__(self, env):
        """
        Initialize the agent with an environment.
        
        Args:
            env: Environment instance (must have num_states and num_actions)
        """
        self.env = env
        self.num_states = env.rows * env.cols
        self.num_actions = len(env.actions)
    
    @abstractmethod
    def choose_action(self, state):
        """
        Select an action based on current state.
        
        Args:
            state: Current state index
            
        Returns:
            action: Selected action index
        """
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """
        Update agent's policy/value function based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode is complete
        """
        pass
    
    @abstractmethod
    def train(self, num_episodes, max_steps):
        """
        Train the agent for specified number of episodes.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            
        Returns:
            Training statistics (implementation-specific)
        """
        pass


# ============================================================================
# FILE: environment.py
# ============================================================================
"""
GridWorld environment for Q-Learning maze solver.
Defines states, actions, rewards, and state transitions.
"""

import numpy as np


class GridWorld:
    """
    2D Grid-World environment for reinforcement learning.
    
    States: Each cell in the grid is a unique state
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Rewards: Defined by REWARD_GOAL, REWARD_WALL, REWARD_STEP in config
    """
    
    def __init__(self, grid_size, start_pos, goal_pos, obstacles, 
                 reward_goal=100, reward_wall=-10, reward_step=-1):
        """
        Initialize the GridWorld environment.
        
        Args:
            grid_size: Tuple (rows, cols) for grid dimensions
            start_pos: Tuple (row, col) for starting position
            goal_pos: Tuple (row, col) for goal position
            obstacles: List of (row, col) tuples marking walls
            reward_goal: Reward for reaching goal
            reward_wall: Penalty for hitting walls/boundaries
            reward_step: Reward for each step (usually negative)
        """
        self.rows, self.cols = grid_size
        self.start = start_pos
        self.goal = goal_pos
        self.current_state = start_pos
        
        # Rewards
        self.reward_goal = reward_goal
        self.reward_wall = reward_wall
        self.reward_step = reward_step
        
        # Define action space: [Up, Down, Left, Right]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        
        # Initialize grid: 0=empty, 1=wall, 2=goal
        self.grid = np.zeros((self.rows, self.cols))
        
        # Mark obstacles
        for obs in obstacles:
            if 0 <= obs[0] < self.rows and 0 <= obs[1] < self.cols:
                self.grid[obs] = 1
        
        # Mark goal
        self.grid[self.goal] = 2
    
    def reset(self):
        """
        Reset environment to starting state.
        
        Returns:
            state_idx: Index of starting state
        """
        self.current_state = self.start
        return self._state_to_index(self.current_state)
    
    def _state_to_index(self, state):
        """
        Convert (row, col) state to single index.
        
        Args:
            state: Tuple (row, col)
            
        Returns:
            index: Integer state index
        """
        return state[0] * self.cols + state[1]
    
    def _index_to_state(self, index):
        """
        Convert single index to (row, col) state.
        
        Args:
            index: Integer state index
            
        Returns:
            state: Tuple (row, col)
        """
        return (index // self.cols, index % self.cols)
    
    def get_valid_actions(self, state):
        """
        Get list of valid actions from given state.
        
        Args:
            state: State index or (row, col) tuple
            
        Returns:
            valid_actions: List of valid action indices
        """
        if isinstance(state, int):
            state = self._index_to_state(state)
        
        valid_actions = []
        for action_idx, delta in enumerate(self.actions):
            next_row = state[0] + delta[0]
            next_col = state[1] + delta[1]
            
            # Check if action leads to valid position
            if (0 <= next_row < self.rows and 
                0 <= next_col < self.cols and 
                self.grid[next_row, next_col] != 1):
                valid_actions.append(action_idx)
        
        return valid_actions
    
    def step(self, action):
        """
        Execute action and return next state, reward, and done flag.
        
        Args:
            action: Integer 0-3 representing movement direction
            
        Returns:
            next_state_idx: Index of next state
            reward: Reward for taking this action
            done: Whether episode is complete (reached goal)
        """
        # Calculate next position
        delta = self.actions[action]
        next_row = self.current_state[0] + delta[0]
        next_col = self.current_state[1] + delta[1]
        
        # Check boundaries
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            # Hit boundary - stay in place, penalty
            return self._state_to_index(self.current_state), self.reward_wall, False
        
        next_state = (next_row, next_col)
        
        # Check if hit wall
        if self.grid[next_state] == 1:
            # Hit wall - stay in place, penalty
            return self._state_to_index(self.current_state), self.reward_wall, False
        
        # Check if reached goal
        if next_state == self.goal:
            self.current_state = next_state
            return self._state_to_index(next_state), self.reward_goal, True
        
        # Valid move to empty cell
        self.current_state = next_state
        return self._state_to_index(next_state), self.reward_step, False
    
    def render(self):
        """
        Render the grid as ASCII art.
        
        Returns:
            grid_str: String representation of the grid
        """
        grid_str = "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.current_state:
                    grid_str += " A "  # Agent
                elif (i, j) == self.goal:
                    grid_str += " G "  # Goal
                elif self.grid[i, j] == 1:
                    grid_str += " # "  # Wall
                elif (i, j) == self.start:
                    grid_str += " S "  # Start
                else:
                    grid_str += " . "  # Empty
            grid_str += "\n"
        return grid_str


# ============================================================================
# FILE: agent_qlearning.py
# ============================================================================
"""
Q-Learning agent implementation.
Learns optimal policy using Q-Learning algorithm with epsilon-greedy exploration.
"""

import numpy as np
# from agent_base import BaseAgent  # Import in actual project


class QLearningAgent(BaseAgent):
    """
    Q-Learning agent for grid-world navigation.
    
    Implements the Q-Learning algorithm:
    Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Uses epsilon-greedy exploration strategy.
    """
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01):
        """
        Initialize Q-Learning agent.
        
        Args:
            env: GridWorld environment instance
            alpha: Learning rate (α)
            gamma: Discount factor (γ)
            epsilon: Initial exploration rate (ε)
            epsilon_decay: Decay multiplier for epsilon per episode
            epsilon_min: Minimum value for epsilon
        """
        super().__init__(env)
        
        # Q-Learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        # Shape: (num_states, num_actions)
        self.q_table = np.zeros((self.num_states, self.num_actions))
    
    def choose_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        
        With probability epsilon: explore (random action)
        With probability 1-epsilon: exploit (best known action)
        
        Args:
            state: Current state index
            
        Returns:
            action: Chosen action index (0-3)
        """
        if np.random.random() < self.epsilon:
            # Explore: choose random action
            return np.random.randint(self.num_actions)
        else:
            # Exploit: choose action with highest Q-value
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether episode terminated
        """
        # Get current Q-value
        current_q = self.q_table[state, action]
        
        # Calculate target Q-value
        if done:
            # Terminal state: no future rewards
            target_q = reward
        else:
            # Non-terminal: reward + discounted max future Q-value
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.gamma * max_next_q
        
        # Update Q-value using learning rate
        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, num_episodes, max_steps, episodes_to_save=None):
        """
        Train the Q-Learning agent for multiple episodes.
        
        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            episodes_to_save: List of episode indices to save paths for
            
        Returns:
            rewards_history: List of total rewards per episode
            steps_history: List of steps taken per episode
            paths_history: Dictionary {episode: path} for saved episodes
        """
        rewards_history = []
        steps_history = []
        paths_history = {}
        
        if episodes_to_save is None:
            episodes_to_save = []
        
        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        print("-" * 60)
        
        for episode in range(num_episodes):
            # Reset environment
            state = self.env.reset()
            total_reward = 0
            steps = 0
            path = [self.env.current_state]
            
            # Run episode
            for step in range(max_steps):
                # Choose and execute action
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                
                # Update Q-table
                self.update(state, action, reward, next_state, done)
                
                # Track statistics
                total_reward += reward
                steps += 1
                path.append(self.env.current_state)
                
                state = next_state
                
                if done:
                    break
            
            # Save path if requested
            if episode in episodes_to_save:
                paths_history[episode] = path
            
            # Decay exploration rate
            self.decay_epsilon()
            
            # Store episode statistics
            rewards_history.append(total_reward)
            steps_history.append(steps)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                print(f"Episode {episode + 1:4d}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:7.2f} | "
                      f"Avg Steps: {avg_steps:6.2f} | "
                      f"Epsilon: {self.epsilon:.4f}")
        
        print("-" * 60)
        print("Training complete!")
        
        return rewards_history, steps_history, paths_history


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


# ============================================================================
# FILE: main.py
# ============================================================================
"""
Main entry point for Q-Learning Grid-World Maze Solver.
Ties together environment, agent, training, and visualization.
"""

# Import all modules
# from config import *
# from environment import GridWorld
# from agent_qlearning import QLearningAgent
# from visualize import (plot_rewards, plot_steps, show_q_heatmap, 
#                        plot_maze_with_path, plot_policy,
#                        create_comprehensive_visualization)


def main():
    """
    Main execution function.
    Sets up environment, trains agent, and generates visualizations.
    """
    print("=" * 70)
    print(" Q-LEARNING GRID-WORLD MAZE SOLVER")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Create Environment
    # ========================================================================
    print("\n[1] Creating Grid-World Environment")
    print("-" * 70)
    
    env = GridWorld(
        grid_size=GRID_SIZE,
        start_pos=START_POS,
        goal_pos=GOAL_POS,
        obstacles=OBSTACLES,
        reward_goal=REWARD_GOAL,
        reward_wall=REWARD_WALL,
        reward_step=REWARD_STEP
    )
    
    print(f"    Grid Size: {GRID_SIZE[0]}x{GRID_SIZE[1]}")
    print(f"    Start Position: {START_POS}")
    print(f"    Goal Position: {GOAL_POS}")
    print(f"    Number of Obstacles: {len(OBSTACLES)}")
    print(f"    Total States: {env.rows * env.cols}")
    print(f"    Actions: {len(env.actions)} (Up, Down, Left, Right)")
    print(f"\n    Reward Structure:")
    print(f"      - Goal: {REWARD_GOAL}")
    print(f"      - Wall/Boundary: {REWARD_WALL}")
    print(f"      - Step: {REWARD_STEP}")
    
    # Print ASCII representation of maze
    print(f"\n    Initial Maze Layout:")
    print(env.render())
    
    # ========================================================================
    # STEP 2: Initialize Q-Learning Agent
    # ========================================================================
    print("\n[2] Initializing Q-Learning Agent")
    print("-" * 70)
    
    agent = QLearningAgent(
        env=env,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN
    )
    
    print(f"    Learning Rate (α): {ALPHA}")
    print(f"    Discount Factor (γ): {GAMMA}")
    print(f"    Initial Exploration Rate (ε): {EPSILON}")
    print(f"    Epsilon Decay: {EPSILON_DECAY}")
    print(f"    Minimum Epsilon: {EPSILON_MIN}")
    print(f"    Q-Table Shape: {agent.q_table.shape}")
    
    # ========================================================================
    # STEP 3: Train Agent
    # ========================================================================
    print("\n[3] Training Agent")
    print("-" * 70)
    
    rewards_history, steps_history, paths_history = agent.train(
        num_episodes=EPISODES,
        max_steps=MAX_STEPS,
        episodes_to_save=SAVE_PATHS_AT
    )
    
    # ========================================================================
    # STEP 4: Display Training Statistics
    # ========================================================================
    print("\n[4] Training Statistics")
    print("-" * 70)
    
    # Calculate statistics
    initial_avg_reward = np.mean(rewards_history[:100])
    final_avg_reward = np.mean(rewards_history[-100:])
    initial_avg_steps = np.mean(steps_history[:100])
    final_avg_steps = np.mean(steps_history[-100:])
    
    print(f"    Initial Performance (first 100 episodes):")
    print(f"      - Average Reward: {initial_avg_reward:.2f}")
    print(f"      - Average Steps: {initial_avg_steps:.2f}")
    print(f"\n    Final Performance (last 100 episodes):")
    print(f"      - Average Reward: {final_avg_reward:.2f}")
    print(f"      - Average Steps: {final_avg_steps:.2f}")
    print(f"\n    Improvement:")
    print(f"      - Reward Gain: {final_avg_reward - initial_avg_reward:+.2f}")
    print(f"      - Steps Reduced: {initial_avg_steps - final_avg_steps:.2f}")
    print(f"\n    Final Epsilon: {agent.epsilon:.4f}")
    
    # ========================================================================
    # STEP 5: Test Learned Policy
    # ========================================================================
    print("\n[5] Testing Learned Policy (Greedy)")
    print("-" * 70)
    
    # Test with pure exploitation (no exploration)
    agent.epsilon = 0.0
    state = env.reset()
    test_path = [env.current_state]
    total_test_reward = 0
    
    for step in range(MAX_STEPS):
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        test_path.append(env.current_state)
        total_test_reward += reward
        if done:
            break
    
    print(f"    Test Episode Results:")
    print(f"      - Steps Taken: {len(test_path) - 1}")
    print(f"      - Total Reward: {total_test_reward}")
    print(f"      - Goal Reached: {env.current_state == env.goal}")
    print(f"      - Optimal Path Length: ~{abs(GOAL_POS[0] - START_POS[0]) + abs(GOAL_POS[1] - START_POS[1])}")
    
    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    print("\n[6] Generating Visualizations")
    print("-" * 70)
    
    # Individual plots
    plot_rewards(rewards_history, window=PLOT_WINDOW)
    plot_steps(steps_history, window=PLOT_WINDOW)
    show_q_heatmap(agent.q_table, env)
    plot_policy(agent.q_table, env)
    
    # Test path visualization
    plot_maze_with_path(env, test_path, 
                       title='Test Episode (Greedy Policy)',
                       save_path='test_path.png')
    
    # Comprehensive visualization
    create_comprehensive_visualization(
        env, agent, rewards_history, steps_history, paths_history
    )
    
    print("\n    Generated Files:")
    print("      ✓ learning_curve.png")
    print("      ✓ steps_curve.png")
    print("      ✓ q_heatmap.png")
    print("      ✓ policy.png")
    print("      ✓ test_path.png")
    print("      ✓ comprehensive_results.png")
    
    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    print("\n[7] Summary")
    print("-" * 70)
    print(f"    ✓ Successfully trained Q-Learning agent for {EPISODES} episodes")
    print(f"    ✓ Agent learned to navigate from {START_POS} to {GOAL_POS}")
    print(f"    ✓ Final performance: {final_avg_steps:.1f} steps (vs {initial_avg_steps:.1f} initially)")
    print(f"    ✓ All visualizations saved successfully")
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print("\n Check the generated PNG files for detailed visualizations.\n")
    
    return env, agent, rewards_history, steps_history, paths_history


if __name__ == "__main__":
    # Execute main training pipeline
    env, agent, rewards, steps, paths = main()
    
    # Optional: Display final plot
    plt.show()


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
HOW TO USE THIS MODULAR PROJECT:
=================================

1. SAVE FILES SEPARATELY:
   Create a directory 'q_learning_maze/' and save each section as:
   - config.py
   - agent_base.py
   - environment.py
   - agent_qlearning.py
   - visualize.py
   - main.py

2. UNCOMMENT IMPORTS:
   In each file, uncomment the import statements at the top

3. RUN THE PROJECT:
   $ cd q_learning_maze/
   $ python main.py

4. CUSTOMIZE:
   Edit config.py to change:
   - Grid size and maze layout
   - Learning hyperparameters
   - Training duration
   - Visualization settings

5. EXTEND:
   - Add new agent types by inheriting from BaseAgent
   - Modify environment.py to create different maze patterns
   - Add new visualization functions in visualize.py

PROJECT STRUCTURE:
==================
q_learning_maze/
│
├── main.py                 # Entry point - runs everything
├── config.py               # All hyperparameters and settings
├── environment.py          # GridWorld environment definition
├── agent_base.py           # Abstract base agent class
├── agent_qlearning.py      # Q-Learning implementation
└── visualize.py            # All plotting functions

OUTPUT FILES:
=============
After running, you'll get:
- learning_curve.png        # Rewards over time
- steps_curve.png           # Efficiency improvement
- q_heatmap.png             # Learned state values
- policy.png                # Optimal actions per state
- test_path.png             # Final greedy policy test
- comprehensive_results.png # All results in one figure

EXAMPLE CUSTOMIZATION:
======================
To create a different maze, edit config.py:

GRID_SIZE = (8, 8)
START_POS = (0, 0)
GOAL_POS = (7, 7)
OBSTACLES = [
    (1, 1), (1, 2), (1, 3),
    (3, 3), (4, 3), (5, 3),
    # Add more obstacles...
]

To try different learning rates, edit config.py:

ALPHA = 0.2        # Faster learning
GAMMA = 0.99       # More far-sighted
EPSILON_DECAY = 0.99  # Slower exploration decay
"""_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curve to '{save_path}'")


def plot_steps(steps_history, window=50, save_path='steps_curve.png'):
    """
    Plot steps per episode showing efficiency improvement.
    
    Args:
        steps_history: List of steps taken per episode
        window: Moving average window size
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 5))
    
    # Calculate moving average
    steps_smooth = np.convolve(steps_history, 
                               np.ones(window)/window, mode='valid')
    
    plt.plot(steps_smooth, linewidth=2, color='#A23B72')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Steps', fontsize=12)
    plt.title(f'Steps to Goal (Moving Average, window={window})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved steps curve to '{save_path}'")


def show_q_heatmap(q_table, env, save_path='q_heatmap.png'):
    """
    Create heatmap showing maximum Q-value for each state.
    
    Args:
        q_table: Q-table array (num_states, num_actions)
        env: GridWorld environment instance
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    # Get maximum Q-value for each state and reshape to grid
    q_max = np.max(q_table, axis=1).reshape(env.rows, env.cols)
    
    # Create heatmap
    im = plt.imshow(q_max, cmap='RdYlGn', aspect='auto')
    plt.colorbar(im, label='Max Q-Value')
    
    # Mark walls and goal
    for i in range(env.rows):
        for j in range(env.cols):
            if env.grid[i, j] == 1:  # Wall
                plt.scatter(j, i, marker='s', s=500, c='black', alpha=0.5)
            elif (i, j) == env.goal:  # Goal
                plt.scatter(j, i, marker='*', s=500, c='gold', edgecolors='black')
            elif (i, j) == env.start:  # Start
                plt.scatter(j, i, marker='o', s=300, c='cyan', edgecolors='black')
    
    plt.title('Learned Q-Values (Max per State)', fontsize=14)
    plt.xlabel('Column', fontsize=12)
    plt.ylabel('Row', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved Q-value heatmap to '{save_path}'")


def plot_maze_with_path(env, path, title='Agent Path', save_path=None):
    """
    Visualize maze with agent's path.
    
    Args:
        env: GridWorld environment instance
        path: List of (row, col) tuples showing agent's path
        title: Plot title
        save_path: Path to save figure (if None, displays instead)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw grid cells
    for i in range(env.rows):
        for j in range(env.cols):
            color = 'white'
            if env.grid[i, j] == 1:  # Wall
                color = 'black'
            elif (i, j) == env.goal:  # Goal
                color = 'gold'
            elif (i, j) == env.start:  # Start
                color = 'lightblue'
            
            rect = patches.Rectangle((j, env.rows - i - 1), 1, 1,
                                    linewidth=1, edgecolor='gray',
                                    facecolor=color)
            ax.add_patch(rect)
    
    # Draw agent path
    if path:
        path_array = np.array(path)
        path_x = path_array[:, 1] + 0.5
        path_y = env.rows - path_array[:, 0] - 0.5
        ax.plot(path_x, path_y, 'r-', linewidth=3, alpha=0.7, label='Path')
        ax.plot(path_x[0], path_y[0], 'go', markersize=12, label='Start')
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=18, label='End')
    
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title(f'{title} ({len(path)} steps)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved maze visualization to '{save_path}'")
    else:
        plt.show()


def plot_policy(q_table, env, save_path='policy.png'):
    """
    Visualize learned policy showing best action per state.
    
    Args:
        q_table: Q-table array
        env: GridWorld environment instance
        save_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid and policy arrows
    for i in range(env.rows):
        for j in range(env.cols):
            if env.grid[i, j] == 1:  # Wall
                rect = patches.Rectangle((j, env.rows - i - 1), 1, 1,
                                        facecolor='black')
                ax.add_patch(rect)
            elif (i, j) == env.goal:  # Goal
                rect = patches.Rectangle((j, env.rows - i - 1), 1, 1,
                                        facecolor='gold')
                ax.add_patch(rect)
            else:
                # Get best action for this state
                state_idx = i * env.cols + j
                best_action = np.argmax(q_table[state_idx])
                
                # Draw arrow indicating best action
                dx, dy = 0, 0
                if best_action == 0:  # Up
                    dx, dy = 0, 0.3
                elif best_action == 1:  # Down
                    dx, dy = 0, -0.3
                elif best_action == 2:  # Left
                    dx, dy = -0.3, 0
                elif best_action == 3:  # Right
                    dx, dy = 0.3, 0
                
                ax.arrow(j + 0.5, env.rows - i - 0.5, dx, dy,
                        head_width=0.2, head_length=0.15, 
                        fc='#2E86AB', ec='#2E86AB', linewidth=2)
    
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect('equal')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    ax.set_title('Learned Policy (Best Action per State)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved policy visualization to '{save_path}'")


def create_comprehensive_visualization(env, agent, rewards_history, 
                                       steps_history, paths_history, 
                                       save_path='comprehensive_results.png'):
    """
    Create a comprehensive multi-panel visualization of all results.
    
    Args:
        env: GridWorld environment
        agent: Trained QLearningAgent
        rewards_history: List of rewards per episode
        steps_history: List of steps per episode
        paths_history: Dictionary of saved paths
        save_path: Path to save the figure
    """
    fig = plt.figure(figsize=(18, 12))
    
    window = 50
    
    # 1. Learning curve
    ax1 = plt.subplot(3, 3, 1)
    rewards_smooth = np.convolve(rewards_history, 
                                 np.ones(window)/window, mode='valid')
    ax1.plot(rewards_smooth, linewidth=2, color='#2E86AB')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Avg Reward')
    ax1.set_title(f'Learning Progress (window={window})')
    ax1.grid(True, alpha=0.3)
    
    # 2. Steps curve
    ax2 = plt.subplot(3, 3, 2)
    steps_smooth = np.convolve(steps_history, 
                               np.ones(window)/window, mode='valid')
    ax2.plot(steps_smooth, linewidth=2, color='#A23B72')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Avg Steps')
    ax2.set_title(f'Steps to Goal (window={window})')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-value heatmap
    ax3 = plt.subplot(3, 3, 3)
    q_max = np.max(agent.q_table, axis=1).reshape(env.rows, env.cols)
    im = ax3.imshow(q_max, cmap='RdYlGn', aspect='auto')
    ax3.set_title('Q-Values (Max per State)')
    plt.colorbar(im, ax=ax3, label='Max Q')
    
    # 4-8. Paths at different episodes
    sorted_episodes = sorted(paths_history.keys())[:5]
    for idx, episode in enumerate(sorted_episodes):
        ax = plt.subplot(3, 3, 4 + idx)
        path = paths_history[episode]
        
        # Draw maze
        for i in range(env.rows):
            for j in range(env.cols):
                color = 'white'
                if env.grid[i, j] == 1:
                    color = 'black'
                elif (i, j) == env.goal:
                    color = 'gold'
                
                rect = patches.Rectangle((j, env.rows - i - 1), 1, 1,
                                        linewidth=0.5, edgecolor='gray',
                                        facecolor=color)
                ax.add_patch(rect)
        
        # Draw path
        path_array = np.array(path)
        path_x = path_array[:, 1] + 0.5
        path_y = env.rows - path_array[:, 0] - 0.5
        ax.plot(path_x, path_y, 'r-', linewidth=2, alpha=0.7)
        ax.plot(path_x[0], path_y[0], 'go', markersize=8)
        ax.plot(path_x[-1], path_y[-1], 'r*', markersize=12)
        
        ax.set_xlim(0, env.cols)
        ax.set_ylim(0, env.rows)
        ax.set_aspect('equal')
        ax.set_title(f'Episode {episode + 1} ({len(path)} steps)')
        ax.grid(True, alpha=0.3)
    
    # 9. Policy visualization
    ax9 = plt.subplot(3, 3, 9)
    for i in range(env.rows):
        for j in range(env.cols):
            if env.grid[i, j] == 1:
                rect = patches.Rectangle((j, env.rows - i - 1), 1, 1,
                                        facecolor='black')
                ax9.add_patch(rect)
            elif (i, j) == env.goal:
                rect = patches.Rectangle((j, env.rows - i - 1), 1, 1,
                                        facecolor='gold')
                ax9.add_patch(rect)
            else:
                state_idx = i * env.cols + j
                best_action = np.argmax(agent.q_table[state_idx])
                
                dx, dy = 0, 0
                if best_action == 0: dx, dy = 0, 0.3
                elif best_action == 1: dx, dy = 0, -0.3
                elif best_action == 2: dx, dy = -0.3, 0
                elif best_action == 3: dx, dy = 0.3, 0
                
                ax9.arrow(j + 0.5, env.rows - i - 0.5, dx, dy,
                         head_width=0.2, head_length=0.15, fc='blue', ec='blue')
    
    ax9.set_xlim(0, env.cols)
    ax9.set_ylim(0, env.rows)
    ax9.set_aspect('equal')
    ax9.set_title('Learned Policy')
    ax9.grid(True, alpha=0.3)
    
    plt.tight
