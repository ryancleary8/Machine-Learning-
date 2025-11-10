# ============================================================================
# FILE: agent_qlearning.py
# ============================================================================
"""
Q-Learning agent implementation.
Learns optimal policy using Q-Learning algorithm with epsilon-greedy exploration.
"""

import numpy as np
from agent_base import BaseAgent  # Import in actual project


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
