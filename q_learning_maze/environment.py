# ============================================================================
# FILE: environment.py
# ============================================================================
"""
GridWorld environment for Q-Learning maze solver.
Defines states, actions, rewards, and state transitions.
"""

import numpy as np


class GridWorld:
    """
    2D Grid-World environment for reinforcement learning.
    
    States: Each cell in the grid is a unique state
    Actions: 0=Up, 1=Down, 2=Left, 3=Right
    Rewards: Defined by REWARD_GOAL, REWARD_WALL, REWARD_STEP in config
    """
    
    def __init__(self, grid_size, start_pos, goal_pos, obstacles,
                 reward_goal=100, reward_wall=-10, reward_step=-1):
        """
        Initialize the GridWorld environment.
        
        Args:
            grid_size: Tuple (rows, cols) for grid dimensions
            start_pos: Tuple (row, col) for starting position
            goal_pos: Tuple (row, col) for goal position
            obstacles: List of (row, col) tuples marking walls
            reward_goal: Reward for reaching goal
            reward_wall: Penalty for hitting walls/boundaries
            reward_step: Reward for each step (usually negative)
        """
        self.rows, self.cols = grid_size
        self.start = start_pos
        self.goal = goal_pos
        self.current_state = start_pos
        
        # Rewards
        self.reward_goal = reward_goal
        self.reward_wall = reward_wall
        self.reward_step = reward_step
        
        # Define action space: [Up, Down, Left, Right]
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.action_names = ['Up', 'Down', 'Left', 'Right']
        
        # Initialize grid: 0=empty, 1=wall, 2=goal
        self.grid = np.zeros((self.rows, self.cols))
        
        # Mark obstacles
        for obs in obstacles:
            if 0 <= obs[0] < self.rows and 0 <= obs[1] < self.cols:
                self.grid[obs] = 1
        
        # Mark goal
        self.grid[self.goal] = 2
    
    def reset(self):
        """
        Reset environment to starting state.
        
        Returns:
            state_idx: Index of starting state
        """
        self.current_state = self.start
        return self._state_to_index(self.current_state)
    
    def _state_to_index(self, state):
        """
        Convert (row, col) state to single index.
        
        Args:
            state: Tuple (row, col)
            
        Returns:
            index: Integer state index
        """
        return state[0] * self.cols + state[1]
    
    def _index_to_state(self, index):
        """
        Convert single index to (row, col) state.
        
        Args:
            index: Integer state index
            
        Returns:
            state: Tuple (row, col)
        """
        return (index // self.cols, index % self.cols)
    
    def get_valid_actions(self, state):
        """
        Get list of valid actions from given state.
        
        Args:
            state: State index or (row, col) tuple
            
        Returns:
            valid_actions: List of valid action indices
        """
        if isinstance(state, int):
            state = self._index_to_state(state)
        
        valid_actions = []
        for action_idx, delta in enumerate(self.actions):
            next_row = state[0] + delta[0]
            next_col = state[1] + delta[1]
            
            # Check if action leads to valid position
            if (0 <= next_row < self.rows and
                0 <= next_col < self.cols and
                self.grid[next_row, next_col] != 1):
                valid_actions.append(action_idx)
        
        return valid_actions
    
    def step(self, action):
        """
        Execute action and return next state, reward, and done flag.
        
        Args:
            action: Integer 0-3 representing movement direction
            
        Returns:
            next_state_idx: Index of next state
            reward: Reward for taking this action
            done: Whether episode is complete (reached goal)
        """
        # Calculate next position
        delta = self.actions[action]
        next_row = self.current_state[0] + delta[0]
        next_col = self.current_state[1] + delta[1]
        
        # Check boundaries
        if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
            # Hit boundary - stay in place, penalty
            return self._state_to_index(self.current_state), self.reward_wall, False
        
        next_state = (next_row, next_col)
        
        # Check if hit wall
        if self.grid[next_state] == 1:
            # Hit wall - stay in place, penalty
            return self._state_to_index(self.current_state), self.reward_wall, False
        
        # Check if reached goal
        if next_state == self.goal:
            self.current_state = next_state
            return self._state_to_index(next_state), self.reward_goal, True
        
        # Valid move to empty cell
        self.current_state = next_state
        return self._state_to_index(next_state), self.reward_step, False
    
    def render(self):
        """
        Render the grid as ASCII art.
        
        Returns:
            grid_str: String representation of the grid
        """
        grid_str = "\n"
        for i in range(self.rows):
            for j in range(self.cols):
                if (i, j) == self.current_state:
                    grid_str += " A "  # Agent
                elif (i, j) == self.goal:
                    grid_str += " G "  # Goal
                elif self.grid[i, j] == 1:
                    grid_str += " # "  # Wall
                elif (i, j) == self.start:
                    grid_str += " S "  # Start
                else:
                    grid_str += " . "  # Empty
            grid_str += "\n"
        return grid_str

