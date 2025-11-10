# agents/agent_base.py - Base class for all agents

from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.training_scores = []
        self.training_steps = []
        
    @abstractmethod
    def choose_action(self, state):
        """Choose an action given the current state"""
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update the agent's knowledge based on experience"""
        pass
    
    def train(self, env, episodes):
        """Train the agent for a number of episodes"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while not env.done and steps < 1000:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            self.training_scores.append(env.score)
            self.training_steps.append(steps)
            
            if (episode + 1) % 10 == 0:
                avg_score = np.mean(self.training_scores[-10:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Score: {avg_score:.2f}, "
                      f"Score: {env.score}, "
                      f"Steps: {steps}")
        
        return self.training_scores, self.training_steps
    
    @abstractmethod
    def save(self, filepath):
        """Save the agent's learned policy"""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """Load a saved policy"""
        pass
