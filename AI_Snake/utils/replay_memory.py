# utils/replay_memory.py - Experience Replay Buffer for DQN

import random
from collections import deque

class ReplayMemory:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """Return current size of memory"""
        return len(self.memory)
