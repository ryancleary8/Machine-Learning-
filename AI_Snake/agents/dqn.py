# agents/agent_dqn.py - Deep Q-Network Agent

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from collections import deque

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
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Q-Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        self.grad_clip_norm = 10.0
        self.learning_starts = 5000
        self.train_every = 4
        self.tau = 0.01  # Polyak averaging coefficient for soft target updates
        self.target_update_steps = None  # set to an int for optional hard updates by step
        
        # Hyperparameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.discount_factor = DISCOUNT_FACTOR
        self.batch_size = BATCH_SIZE
        self.target_update_freq = TARGET_UPDATE_FREQ
        
        self.eps_start = EPSILON_START
        self.eps_end = EPSILON_MIN
        self.eps_decay_frames = 100000
        
        self.training_scores = []
        self.training_steps = []
        self.steps_done = 0
        
    def current_epsilon(self):
        t = min(self.steps_done, self.eps_decay_frames)
        frac = 1.0 - (t / self.eps_decay_frames)
        return self.eps_end + (self.eps_start - self.eps_end) * frac

    def soft_update(self, tau=None):
        """Soft-update target net towards policy net (Polyak averaging)."""
        if tau is None:
            tau = self.tau
        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)

    def choose_action(self, state):
        """Epsilon-greedy action selection with step-based epsilon and eval/train modes."""
        eps = self.current_epsilon()
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)

        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        self.policy_net.train()
        return action
    
    def update(self, state, action, reward, next_state, done):
        """Store transition and train network (Double DQN, Huber loss, soft target updates)."""
        # Optional reward clipping for stability
        if reward is not None:
            reward = float(max(min(reward, 1.0), -1.0))

        # Store transition in replay memory
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1

        # Gating: wait for sufficient experience and train every N steps
        if len(self.memory) < self.learning_starts or (self.steps_done % self.train_every) != 0:
            return

        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states      = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions     = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)   # [B,1]
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)            # [B]
        next_states = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones       = torch.as_tensor(dones, dtype=torch.float32, device=self.device)              # [B]

        # Current Q(s,a)
        q_pred = self.policy_net(states).gather(1, actions).squeeze(1)  # [B]

        # Double DQN target: select via policy net, evaluate via target net
        with torch.no_grad():
            next_q_policy = self.policy_net(next_states)                               # [B, A]
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)                   # [B,1]
            next_q_target = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # [B]
            q_target = rewards + (1.0 - dones) * self.discount_factor * next_q_target # [B]

        # Loss (Huber)
        loss = self.criterion(q_pred, q_target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        # Target network updates
        self.soft_update(self.tau)  # soft update every optimization step

        # Optional hard update by steps if configured
        if self.target_update_steps is not None and (self.steps_done % self.target_update_steps) == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
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
            
            self.training_scores.append(env.score)
            self.training_steps.append(steps)
            
            if (episode + 1) % PLOT_INTERVAL == 0:
                avg_score = np.mean(self.training_scores[-PLOT_INTERVAL:])
                print(f"Episode {episode + 1}/{episodes}, "
                      f"Avg Score: {avg_score:.2f}, "
                      f"Score: {env.score}, "
                      f"Steps: {steps}, "
                      f"Epsilon: {self.current_epsilon():.3f}, "
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
