# agents/agent_qlearning.py - Q-Learning Agent

import numpy as np
import random
import pickle
from config import *

class QLearningAgent:
    """Q-Learning agent using tabular Q-table"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        
        # Hyperparameters
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        
        self.training_scores = []
        self.training_steps = []
        
    def _get_state_key(self, state):
        """Convert state array to hashable tuple"""
        return tuple(state)
    
    def _get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        return self.q_table[state_key][action]
    
    def _set_q_value(self, state, action, value):
        """Set Q-value for state-action pair"""
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        self.q_table[state_key][action] = value
    
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        # Greedy action
        state_key = self._get_state_key(state)
        if state_key not in self.q_table:
            return random.randint(0, self.action_size - 1)
        
        return np.argmax(self.q_table[state_key])
    
    def update(self, state, action, reward, next_state, done):
        """Q-Learning update rule"""
        current_q = self._get_q_value(state, action)
        
        if done:
            target_q = reward
        else:
            next_state_key = self._get_state_key(next_state)
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            max_next_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.discount_factor * max_next_q
        
        # Q-Learning update
        new_q = current_q + self.learning_rate * (target_q - current_q)
        self._set_q_value(state, action, new_q)
    
    def train(self, env, episodes):
        """Train the agent"""
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while not env.done and steps < MAX_STEPS_PER_EPISODE:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            self.training_scores.append(env.score)
            self.training_steps.append(steps)
            
            if (episode + 1) % PLOT_INTERVAL == 0:
                avg_score = np.mean(self.training_scores[-PLOT_INTERVAL:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Score: {avg_score:.2f}, "
                      f"Score: {env.score}, "
                      f"Steps: {steps}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Q-Table Size: {len(self.q_table)}")
        
        return self.training_scores, self.training_steps
    
    def save(self, filepath):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {filepath}")
    
    def load(self, filepath):
        """Load Q-table from file"""
        with open(filepath, 'rb') as f:
            self.q_table = pickle.load(f)
        print(f"Q-table loaded from {filepath}")
