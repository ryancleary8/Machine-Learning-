# ============================================================================
# FILE: agent_sarsa.py
# ============================================================================
"""SARSA agent implementation for the grid-world maze."""

import numpy as np

from agent_base import BaseAgent


class SARSAAgent(BaseAgent):
    """On-policy SARSA agent with epsilon-greedy exploration."""

    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(env)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((self.num_states, self.num_actions))

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done, next_action=None):
        current_q = self.q_table[state, action]

        if done:
            target_q = reward
        else:
            if next_action is None:
                next_action = np.argmax(self.q_table[next_state])
            target_q = reward + self.gamma * self.q_table[next_state, next_action]

        self.q_table[state, action] = current_q + self.alpha * (target_q - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes, max_steps, episodes_to_save=None):
        rewards_history = []
        steps_history = []
        paths_history = {}

        if episodes_to_save is None:
            episodes_to_save = []

        print(f"Training SARSA agent for {num_episodes} episodes...")
        print("-" * 60)

        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            total_reward = 0
            steps = 0
            path = [self.env.current_state]

            for _ in range(max_steps):
                next_state, reward, done = self.env.step(action)
                path.append(self.env.current_state)

                next_action = self.choose_action(next_state) if not done else None

                self.update(state, action, reward, next_state, done,
                            next_action=next_action)

                total_reward += reward
                steps += 1

                state = next_state
                action = next_action if next_action is not None else action

                if done:
                    break

            if episode in episodes_to_save:
                paths_history[episode] = path

            self.decay_epsilon()

            rewards_history.append(total_reward)
            steps_history.append(steps)

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
