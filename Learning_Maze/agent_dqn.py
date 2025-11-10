# =============================================================================
# FILE: agent_dqn.py
# =============================================================================
"""Deep Q-Network agents for the grid-world maze."""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent_base import BaseAgent


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.int64),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.int64),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.memory)


class DQNNetwork(nn.Module):
    """Feed-forward network for approximating Q-values."""

    def __init__(self, input_dim, output_dim, hidden_units):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DQNAgent(BaseAgent):
    """Vanilla Deep Q-Network agent."""

    def __init__(
        self,
        env,
        alpha=1e-3,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        buffer_size=5000,
        target_update=20,
        hidden_units=(128, 128),
    ):
        super().__init__(env)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input_dim = self.num_states
        output_dim = self.num_actions

        self.policy_net = DQNNetwork(input_dim, output_dim, hidden_units).to(self.device)
        self.target_net = DQNNetwork(input_dim, output_dim, hidden_units).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(buffer_size)
        self.training_steps = 0

        # Pre-build identity matrix for one-hot encoding states
        self.eye = torch.eye(self.num_states, device=self.device)

        # Keep a numpy Q-table for compatibility with visualisations
        self.q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def _state_tensor(self, state_indices):
        if isinstance(state_indices, int):
            return self.eye[state_indices].unsqueeze(0)
        if isinstance(state_indices, np.ndarray):
            indices = torch.from_numpy(state_indices).long().to(self.device)
        else:
            indices = torch.tensor(state_indices, dtype=torch.long, device=self.device)
        return self.eye[indices]

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_tensor = self._state_tensor(state)
            q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_tensor = self._state_tensor(states)
        next_state_tensor = self._state_tensor(next_states)
        action_tensor = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        reward_tensor = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        done_tensor = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_tensor).gather(1, action_tensor)

        with torch.no_grad():
            next_q_values = self.target_net(next_state_tensor).max(1, keepdim=True)[0]
            target_values = reward_tensor + (1 - done_tensor) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _refresh_q_table(self):
        with torch.no_grad():
            full_state_tensor = self.eye
            q_values = self.policy_net(full_state_tensor).cpu().numpy()
        self.q_table = q_values

    def train(self, num_episodes, max_steps, episodes_to_save=None):
        rewards_history = []
        steps_history = []
        paths_history = {}

        if episodes_to_save is None:
            episodes_to_save = []

        print(f"Training DQN agent for {num_episodes} episodes...")
        print("-" * 60)

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            path = [self.env.current_state]

            for _ in range(max_steps):
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                self.update(state, action, reward, next_state, done)

                total_reward += reward
                steps += 1
                path.append(self.env.current_state)

                state = next_state

                if done:
                    break

            if episode in episodes_to_save:
                paths_history[episode] = path

            self.decay_epsilon()

            rewards_history.append(total_reward)
            steps_history.append(steps)

            if (episode + 1) % self.target_update == 0:
                self._refresh_q_table()

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                avg_steps = np.mean(steps_history[-100:])
                print(
                    f"Episode {episode + 1:4d}/{num_episodes} | "
                    f"Avg Reward: {avg_reward:7.2f} | "
                    f"Avg Steps: {avg_steps:6.2f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        # Final refresh for visualisation
        self._refresh_q_table()

        print("-" * 60)
        print("Training complete!")

        return rewards_history, steps_history, paths_history


class DoubleDQNAgent(DQNAgent):
    """Double DQN variant that reduces overestimation bias."""

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        state_tensor = self._state_tensor(states)
        next_state_tensor = self._state_tensor(next_states)
        action_tensor = torch.from_numpy(actions).long().unsqueeze(1).to(self.device)
        reward_tensor = torch.from_numpy(rewards).float().unsqueeze(1).to(self.device)
        done_tensor = torch.from_numpy(dones).float().unsqueeze(1).to(self.device)

        q_values = self.policy_net(state_tensor).gather(1, action_tensor)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_tensor).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_state_tensor).gather(1, next_actions)
            target_values = reward_tensor + (1 - done_tensor) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_steps += 1

        if self.training_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
