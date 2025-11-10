"""BestAgent: Adaptive Snake agent combining reinforcement learning and safety-aware pathfinding."""

from collections import deque
import random
import pickle
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from config import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    MAX_STEPS_PER_EPISODE,
    PLOT_INTERVAL,
)
from .agent_base import BaseAgent

Position = Tuple[int, int]


class BestAgent(BaseAgent):
    """Adaptive agent that augments Q-learning with safety heuristics and pathfinding."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__(state_size, action_size)
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.env = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_environment(self, env) -> None:
        """Attach the current environment so that heuristics can inspect it."""

        self.env = env

    # Q-learning -----------------------------------------------------------------
    def choose_action(self, state: np.ndarray) -> int:
        """Return an action using an epsilon-greedy policy with safety overrides."""

        if self.env is None:
            raise RuntimeError("Environment must be set via set_environment before acting.")

        state_key = self._state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)

        if random.random() < self.epsilon:
            safe_actions = self._safe_actions(range(self.action_size))
            if safe_actions:
                return random.choice(safe_actions)
            return random.randrange(self.action_size)

        # Exploit learned policy
        greedy_action = int(np.argmax(self.q_table[state_key]))
        action = greedy_action

        if not self._is_action_safe(action):
            path = self._find_safe_path_to_apple()
            if path:
                action = self._direction_to_action(path[0])
            else:
                tail_action = self._move_toward_tail()
                if tail_action is not None:
                    action = tail_action
                else:
                    safe_actions = self._safe_actions(range(self.action_size))
                    if safe_actions:
                        action = random.choice(safe_actions)

        return action

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """Standard Q-learning update."""

        state_key = self._state_key(state)
        next_state_key = self._state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)

        current_q = self.q_table[state_key][action]
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * float(np.max(self.q_table[next_state_key]))

        self.q_table[state_key][action] = current_q + self.learning_rate * (target - current_q)

    def train(self, env, episodes: int):
        """Train the agent in the provided environment."""

        self.set_environment(env)
        for episode in range(episodes):
            state = env.reset()
            steps = 0

            while not env.done and steps < MAX_STEPS_PER_EPISODE:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                steps += 1

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.training_scores.append(env.score)
            self.training_steps.append(steps)

            if (episode + 1) % PLOT_INTERVAL == 0:
                avg_score = float(np.mean(self.training_scores[-PLOT_INTERVAL:]))
                print(
                    f"Episode {episode + 1}/{episodes}, Avg Score: {avg_score:.2f}, "
                    f"Score: {env.score}, Steps: {steps}, Epsilon: {self.epsilon:.3f}, "
                    f"Q-table entries: {len(self.q_table)}"
                )

        return self.training_scores, self.training_steps

    def save(self, filepath: str) -> None:
        """Persist the learned Q-table."""

        with open(filepath, "wb") as fh:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon,
                },
                fh,
            )

    def load(self, filepath: str) -> None:
        """Load a previously saved Q-table."""

        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)
        self.q_table = payload.get("q_table", {})
        self.epsilon = payload.get("epsilon", 0.0)

    # ------------------------------------------------------------------
    # Safety helpers and pathfinding
    # ------------------------------------------------------------------
    def _state_key(self, state: np.ndarray) -> Tuple[int, ...]:
        return tuple(int(v) for v in state.tolist())

    def _safe_actions(self, actions: Iterable[int]) -> List[int]:
        return [action for action in actions if self._is_action_safe(action)]

    def _is_action_safe(self, action: int) -> bool:
        env = self.env
        if env is None:
            return True

        next_direction = self._direction_after_action(env.direction, action)
        next_head = self._next_position(env.snake[0], next_direction)

        if self._hits_wall(next_head, env.grid_size):
            return False
        if next_head in env.snake:
            return False

        new_body = list(env.snake)
        new_body.insert(0, next_head)
        if next_head != env.food:
            new_body.pop()

        tail = new_body[-1]
        blocked = set(new_body[:-1])
        return self._bfs_path_exists(next_head, tail, env.grid_size, blocked)

    def _find_safe_path_to_apple(self) -> Optional[List[Position]]:
        env = self.env
        if env is None:
            return None

        blocked = set(env.snake)
        blocked.discard(env.snake[-1])
        path = self._bfs_path(env.snake[0], env.food, env.grid_size, blocked)
        if not path:
            return None

        simulated_body = list(env.snake)
        new_food_consumed = False
        for step in path:
            simulated_body.insert(0, step)
            if step == env.food:
                # snake grows, do not remove tail this move
                new_food_consumed = True
            else:
                simulated_body.pop()
                new_food_consumed = False

            if new_food_consumed:
                break

        new_head = simulated_body[0]
        tail = simulated_body[-1]
        blocked_after = set(simulated_body[:-1])
        if self._bfs_path_exists(new_head, tail, env.grid_size, blocked_after):
            return path
        return None

    def _move_toward_tail(self) -> Optional[int]:
        env = self.env
        if env is None:
            return None

        head = env.snake[0]
        tail = env.snake[-1]
        blocked = set(env.snake[:-1])
        path = self._bfs_path(head, tail, env.grid_size, blocked)
        if path:
            return self._direction_to_action(path[0])

        safe = self._safe_actions(range(self.action_size))
        if safe:
            return random.choice(safe)
        return None

    # ------------------------------------------------------------------
    # Direction utilities
    # ------------------------------------------------------------------
    def _direction_after_action(self, direction: str, action: int) -> str:
        order = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx = order.index(direction)
        if action == 1:  # left
            return order[(idx - 1) % 4]
        if action == 2:  # right
            return order[(idx + 1) % 4]
        return direction

    def _direction_to_action(self, next_pos: Position) -> int:
        env = self.env
        if env is None:
            return 0
        head = env.snake[0]
        target_direction = self._direction_from_positions(head, next_pos)
        if target_direction is None:
            return 0
        current_direction = env.direction
        order = ["UP", "RIGHT", "DOWN", "LEFT"]
        idx_current = order.index(current_direction)
        idx_target = order.index(target_direction)
        if idx_target == idx_current:
            return 0
        if idx_target == (idx_current - 1) % 4:
            return 1
        if idx_target == (idx_current + 1) % 4:
            return 2
        return 0

    def _direction_from_positions(self, start: Position, end: Position) -> Optional[str]:
        if end[0] == start[0] - 1 and end[1] == start[1]:
            return "UP"
        if end[0] == start[0] + 1 and end[1] == start[1]:
            return "DOWN"
        if end[0] == start[0] and end[1] == start[1] - 1:
            return "LEFT"
        if end[0] == start[0] and end[1] == start[1] + 1:
            return "RIGHT"
        return None

    def _next_position(self, position: Position, direction: str) -> Position:
        x, y = position
        if direction == "UP":
            return (x - 1, y)
        if direction == "DOWN":
            return (x + 1, y)
        if direction == "LEFT":
            return (x, y - 1)
        if direction == "RIGHT":
            return (x, y + 1)
        return position

    def _hits_wall(self, position: Position, grid_size: int) -> bool:
        x, y = position
        return x < 0 or x >= grid_size or y < 0 or y >= grid_size

    # ------------------------------------------------------------------
    # Graph search utilities
    # ------------------------------------------------------------------
    def _neighbors(self, position: Position, grid_size: int) -> Iterable[Position]:
        for direction in ("UP", "DOWN", "LEFT", "RIGHT"):
            nxt = self._next_position(position, direction)
            if not self._hits_wall(nxt, grid_size):
                yield nxt

    def _bfs_path(
        self,
        start: Position,
        goal: Position,
        grid_size: int,
        blocked: Set[Position],
    ) -> List[Position]:
        if start == goal:
            return []

        queue: Deque[Tuple[Position, List[Position]]] = deque()
        queue.append((start, []))
        visited: Set[Position] = {start}

        while queue:
            current, path = queue.popleft()
            for neighbor in self._neighbors(current, grid_size):
                if neighbor in visited or neighbor in blocked:
                    continue
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        return []

    def _bfs_path_exists(
        self,
        start: Position,
        goal: Position,
        grid_size: int,
        blocked: Set[Position],
    ) -> bool:
        if start == goal:
            return True
        queue: Deque[Position] = deque([start])
        visited: Set[Position] = {start}

        while queue:
            current = queue.popleft()
            for neighbor in self._neighbors(current, grid_size):
                if neighbor in visited or neighbor in blocked:
                    continue
                if neighbor == goal:
                    return True
                visited.add(neighbor)
                queue.append(neighbor)
        return False
