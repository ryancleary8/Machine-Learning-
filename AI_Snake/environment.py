# environment.py - Snake Game Environment

import numpy as np
import random
from config import *

class SnakeEnvironment:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.reset()
        
    def reset(self):
        """Reset the game to initial state"""
        # Snake starts in the middle
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])
        
        # Place food randomly
        self.food = self._place_food()
        
        self.score = 0
        self.steps = 0
        self.done = False
        
        return self.get_state()
    
    def _place_food(self):
        """Place food in a random empty location"""
        while True:
            food = (random.randint(0, self.grid_size - 1), 
                   random.randint(0, self.grid_size - 1))
            if food not in self.snake:
                return food
    
    def get_state(self):
        """
        Return state representation for the agent
        State includes:
        - Danger in each direction (straight, left, right)
        - Direction the snake is moving
        - Food location relative to head
        """
        head = self.snake[0]
        
        # Danger detection
        danger_straight = self._is_collision(self._get_next_position(head, self.direction))
        danger_left = self._is_collision(self._get_next_position(head, self._turn_left()))
        danger_right = self._is_collision(self._get_next_position(head, self._turn_right()))
        
        # Direction encoding (one-hot)
        dir_up = self.direction == 'UP'
        dir_down = self.direction == 'DOWN'
        dir_left = self.direction == 'LEFT'
        dir_right = self.direction == 'RIGHT'
        
        # Food location relative to head
        food_up = self.food[0] < head[0]
        food_down = self.food[0] > head[0]
        food_left = self.food[1] < head[1]
        food_right = self.food[1] > head[1]
        
        state = [
            danger_straight, danger_left, danger_right,
            dir_up, dir_down, dir_left, dir_right,
            food_up, food_down, food_left, food_right
        ]
        
        return np.array(state, dtype=int)
    
    def _get_next_position(self, pos, direction):
        """Get the next position given current position and direction"""
        x, y = pos
        if direction == 'UP':
            return (x - 1, y)
        elif direction == 'DOWN':
            return (x + 1, y)
        elif direction == 'LEFT':
            return (x, y - 1)
        elif direction == 'RIGHT':
            return (x, y + 1)
        return pos
    
    def _is_collision(self, pos):
        """Check if position results in collision"""
        x, y = pos
        # Wall collision
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        # Self collision
        if pos in self.snake:
            return True
        return False
    
    def _turn_left(self):
        """Return the direction if turning left"""
        turns = {'UP': 'LEFT', 'LEFT': 'DOWN', 'DOWN': 'RIGHT', 'RIGHT': 'UP'}
        return turns[self.direction]
    
    def _turn_right(self):
        """Return the direction if turning right"""
        turns = {'UP': 'RIGHT', 'RIGHT': 'DOWN', 'DOWN': 'LEFT', 'LEFT': 'UP'}
        return turns[self.direction]
    
    def step(self, action):
        """
        Execute action and return next_state, reward, done
        Actions: 0 = straight, 1 = left, 2 = right
        """
        self.steps += 1
        
        # Update direction based on action
        if action == 1:  # Turn left
            self.direction = self._turn_left()
        elif action == 2:  # Turn right
            self.direction = self._turn_right()
        # action == 0 means continue straight
        
        # Move snake
        head = self.snake[0]
        new_head = self._get_next_position(head, self.direction)
        
        # Check collision
        if self._is_collision(new_head):
            self.done = True
            return self.get_state(), REWARD_DEATH, True
        
        # Add new head
        self.snake.insert(0, new_head)
        
        # Check if food eaten
        reward = REWARD_STEP
        if new_head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self.food = self._place_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
        
        # Check if max steps reached
        if self.steps >= MAX_STEPS_PER_EPISODE:
            self.done = True
        
        return self.get_state(), reward, self.done
    
    def render(self):
        """Return a visual representation of the current state"""
        grid = np.zeros((self.grid_size, self.grid_size))
        
        # Mark snake body
        for segment in self.snake[1:]:
            grid[segment] = 1
        
        # Mark snake head
        grid[self.snake[0]] = 2
        
        # Mark food
        grid[self.food] = 3
        
        return grid
