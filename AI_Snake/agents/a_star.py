"""A* pathfinding agent for the Snake environment."""

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    MAX_STEPS_PER_EPISODE,
    PLOT_INTERVAL,
)


class AStarAgent:
    """Deterministic agent that follows an A* path to the food."""

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.environment = None

        # Training statistics (used for compatibility with the training loop)
        self.training_scores: List[int] = []
        self.training_steps: List[int] = []

        # Exploration parameters maintained for interface compatibility
        self.epsilon = 0.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 1.0

    # ------------------------------------------------------------------
    # Interface helpers
    # ------------------------------------------------------------------
    def set_environment(self, env) -> None:
        """Attach the environment instance so the agent can access the grid."""
        self.environment = env

    # ------------------------------------------------------------------
    # Core agent logic
    # ------------------------------------------------------------------
    def choose_action(self, state, env=None):  # pylint: disable=unused-argument
        """Select the next action using A* pathfinding."""
        if self.environment is None:
            raise ValueError("Environment must be attached via set_environment before use.")

        next_direction = self._get_next_direction()
        if next_direction is None:
            # Fall back to a safe move if no path exists
            next_direction = self.environment.direction

        # Map desired direction to action (0=straight,1=left,2=right)
        return self._direction_to_action(next_direction)

    def update(self, state, action, reward, next_state, done):  # pylint: disable=unused-argument
        """A* agent is not learning-based, so no update is required."""
        return

    def train(self, env, episodes):
        """Run episodes to collect statistics (agent itself does not learn)."""
        self.set_environment(env)

        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            steps = 0

            while not env.done and steps < MAX_STEPS_PER_EPISODE:
                action = self.choose_action(state)
                state, reward, done = env.step(action)
                self.update(None, None, None, None, None)
                total_reward += reward
                steps += 1

            self.training_scores.append(env.score)
            self.training_steps.append(steps)

            if (episode + 1) % PLOT_INTERVAL == 0:
                avg_score = np.mean(self.training_scores[-PLOT_INTERVAL:])
                print(
                    f"Episode {episode + 1}/{episodes}, "
                    f"Avg Score: {avg_score:.2f}, Score: {env.score}, Steps: {steps}"
                )

        return self.training_scores, self.training_steps

    # ------------------------------------------------------------------
    # Save / Load - deterministic agent so nothing to persist
    # ------------------------------------------------------------------
    def save(self, filepath):
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write("AStarAgent does not maintain persistent state.\n")
        print(f"A* agent metadata saved to {filepath}")

    def load(self, filepath):
        # Nothing to load; ensure file exists for symmetry with save
        try:
            with open(filepath, "r", encoding="utf-8") as handle:
                handle.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "No saved state for AStarAgent. Please ensure the file exists."
            ) from exc
        print(f"A* agent metadata loaded from {filepath}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_next_direction(self) -> Optional[str]:
        env = self.environment
        head = env.snake[0]
        food = env.food

        path = self._a_star(head, food)
        if len(path) < 2:
            return None
        next_cell = path[1]
        hx, hy = head
        nx, ny = next_cell
        if nx < hx:
            return "UP"
        if nx > hx:
            return "DOWN"
        if ny < hy:
            return "LEFT"
        if ny > hy:
            return "RIGHT"
        return None

    def _direction_to_action(self, desired_direction: str) -> int:
        current_direction = self.environment.direction
        if desired_direction == current_direction:
            return 0
        if desired_direction == self.environment._turn_left():  # pylint: disable=protected-access
            return 1
        if desired_direction == self.environment._turn_right():  # pylint: disable=protected-access
            return 2

        # If the desired direction is the opposite (illegal), choose any safe direction
        for action, direction in enumerate(
            [
                current_direction,
                self.environment._turn_left(),  # pylint: disable=protected-access
                self.environment._turn_right(),  # pylint: disable=protected-access
            ]
        ):
            next_pos = self.environment._get_next_position(  # pylint: disable=protected-access
                self.environment.snake[0], direction
            )
            if not self.environment._is_collision(next_pos):  # pylint: disable=protected-access
                return action
        return 0

    def _a_star(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute the A* path from start to goal avoiding the snake body."""
        env = self.environment
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))

        return [start]

    def _neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        env = self.environment
        x, y = cell
        candidate_neighbors = [
            (x - 1, y),
            (x + 1, y),
            (x, y - 1),
            (x, y + 1),
        ]

        snake_body = set(env.snake[:-1])  # allow movement into the tail cell
        valid_neighbors = []
        for nx, ny in candidate_neighbors:
            if 0 <= nx < env.grid_size and 0 <= ny < env.grid_size:
                if (nx, ny) not in snake_body:
                    valid_neighbors.append((nx, ny))
        return valid_neighbors

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
