"""Proximal Policy Optimization (PPO) agent implementation."""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    DISCOUNT_FACTOR,
    LEARNING_RATE,
    MAX_STEPS_PER_EPISODE,
    PLOT_INTERVAL,
    PPO_BATCH_SIZE,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
)

from utils.ppo_model import ActorCritic


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool


class PPOAgent:
    """Actor-critic agent trained with PPO."""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

        # Hyperparameters
        self.gamma = DISCOUNT_FACTOR
        self.lam = PPO_GAE_LAMBDA
        self.clip_epsilon = PPO_CLIP_EPS
        self.epochs = PPO_EPOCHS
        self.batch_size = PPO_BATCH_SIZE
        self.entropy_coef = PPO_ENTROPY_COEF

        # Experience buffer
        self.memory: List[Transition] = []

        # Training stats for compatibility with visualisation logic
        self.training_scores: List[int] = []
        self.training_steps: List[int] = []

        # Attributes used by the main loop for epsilon decay (not used in PPO)
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0

    # ------------------------------------------------------------------
    def choose_action(self, state, env=None):  # pylint: disable=unused-argument
        state_array = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_array).unsqueeze(0).to(self.device)

        with torch.no_grad():
            distribution, value = self.policy(state_tensor)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        transition = Transition(
            state=state_array,
            action=int(action.item()),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
            reward=0.0,
            done=False,
        )
        self.memory.append(transition)
        return int(action.item())

    def update(self, state, action, reward, next_state, done):  # pylint: disable=unused-argument
        if not self.memory:
            return
        self.memory[-1].reward = reward
        self.memory[-1].done = done

        if done:
            self._optimize_model()

    def _optimize_model(self):
        if not self.memory:
            return

        states = torch.tensor([t.state for t in self.memory], dtype=torch.float32, device=self.device)
        actions = torch.tensor([t.action for t in self.memory], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t.log_prob for t in self.memory], dtype=torch.float32, device=self.device)
        values = torch.tensor([t.value for t in self.memory], dtype=torch.float32, device=self.device)
        rewards = [t.reward for t in self.memory]
        dones = [t.done for t in self.memory]

        returns, advantages = self._compute_gae(rewards, dones, values)

        dataset_size = states.size(0)
        for _ in range(self.epochs):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start_idx in range(0, dataset_size, self.batch_size):
                idx = permutation[start_idx : start_idx + self.batch_size]
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_returns = returns[idx]
                batch_advantages = advantages[idx]

                distribution, new_values = self.policy(batch_states)
                new_log_probs = distribution.log_prob(batch_actions)
                entropy = distribution.entropy().mean()

                ratio = (new_log_probs - batch_old_log_probs).exp()
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(new_values.squeeze(-1), batch_returns)

                loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

        self.memory.clear()

    def _compute_gae(self, rewards, dones, values):
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0

        for step in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + self.gamma * next_value * mask - values[step]
            gae = delta + self.gamma * self.lam * mask * gae
            returns.insert(0, gae + values[step])
            advantages.insert(0, gae)
            next_value = values[step]

        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ------------------------------------------------------------------
    def train(self, env, episodes):
        scores = []
        steps_history = []

        for episode in range(episodes):
            state = env.reset()
            score = 0
            steps = 0
            self.memory.clear()

            while not env.done and steps < MAX_STEPS_PER_EPISODE:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)

                state = next_state
                score += reward
                steps += 1

            scores.append(env.score)
            steps_history.append(steps)

            if (episode + 1) % PLOT_INTERVAL == 0:
                avg_score = np.mean(scores[-PLOT_INTERVAL:])
                print(
                    f"Episode {episode + 1}/{episodes}, Avg Score: {avg_score:.2f}, "
                    f"Score: {env.score}, Steps: {steps}"
                )

        self.training_scores = scores
        self.training_steps = steps_history
        return scores, steps_history

    # ------------------------------------------------------------------
    def save(self, filepath):
        torch.save(self.policy.state_dict(), filepath)
        print(f"PPO model saved to {filepath}")

    def load(self, filepath):
        state_dict = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        print(f"PPO model loaded from {filepath}")
