# agents/agent_dqn.py - Deep Q-Network Agent

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from config import *

# Import neural network and replay memory
import sys
sys.path.append('..')
from utils.dqn_model import DQNNetwork
from utils.replay_memory import ReplayMemory

class DQNAgent:
    """Deep Q-Network agent with experience replay"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Q-Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Hyperparameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.discount_factor = DISCOUNT_FACTOR
        self.batch_size = BATCH_SIZE
        self.target_update_freq = TARGET_UPDATE_FREQ
        
        self.training_scores = []
        self.training_steps = []
        self.steps_done = 0
        
    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        """Store transition and train network"""
        # Store transition in replay memory
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1
        
        # Only train if enough samples in memory
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
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
            
            # Update target network
            if (episode + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
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
                      f"Memory Size: {len(self.memory)}")
        
        return self.training_scores, self.training_steps
    
    def save(self, filepath):
        """Save model to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")
