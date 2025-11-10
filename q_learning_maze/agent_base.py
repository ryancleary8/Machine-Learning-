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
