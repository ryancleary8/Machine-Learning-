# visualize.py - Visualization for Snake AI

import pygame
import numpy as np
import matplotlib.pyplot as plt
from config import *

class SnakeVisualizer:
    """Handles visualization of Snake game and training progress"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.cell_size = CELL_SIZE
        self.width = grid_size * self.cell_size
        self.height = grid_size * self.cell_size
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.DARK_GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        
        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f'Snake AI - {ALGORITHM} - {DIFFICULTY}')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
    def render(self, env, score, episode=None):
        """Render the current game state"""
        self.screen.fill(self.BLACK)
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                 self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, self.GRAY, rect, 1)
        
        # Draw food
        food_rect = pygame.Rect(env.food[1] * self.cell_size, 
                               env.food[0] * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.RED, food_rect)
        
        # Draw snake body
        for segment in env.snake[1:]:
            snake_rect = pygame.Rect(segment[1] * self.cell_size,
                                    segment[0] * self.cell_size,
                                    self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.DARK_GREEN, snake_rect)
        
        # Draw snake head
        head = env.snake[0]
        head_rect = pygame.Rect(head[1] * self.cell_size,
                               head[0] * self.cell_size,
                               self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.GREEN, head_rect)
        
        # Draw score
        score_text = self.font.render(f'Score: {score}', True, self.WHITE)
        self.screen.blit(score_text, (10, 10))
        
        if episode is not None:
            episode_text = self.font.render(f'Episode: {episode}', True, self.WHITE)
            self.screen.blit(episode_text, (10, 50))
        
        pygame.display.flip()
        self.clock.tick(GAME_SPEED)
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        return True
    
    def close(self):
        """Close the pygame window"""
        pygame.quit()

def plot_training_progress(scores, steps, algorithm_name):
    """Plot training progress graphs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot scores
    ax1.plot(scores, alpha=0.6, label='Score per Episode')
    
    # Calculate and plot moving average
    window = 20
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(scores)), moving_avg, 
                linewidth=2, label=f'{window}-Episode Moving Avg', color='red')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{algorithm_name} - Training Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot steps per episode
    ax2.plot(steps, alpha=0.6, label='Steps per Episode', color='orange')
    
    if len(steps) >= window:
        moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(steps)), moving_avg_steps,
                linewidth=2, label=f'{window}-Episode Moving Avg', color='red')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title(f'{algorithm_name} - Steps per Episode')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_{algorithm_name}_{DIFFICULTY}.png', dpi=150)
    plt.show()

def compare_algorithms(results_dict):
    """Compare performance of multiple algorithms"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    window = 20
    
    for algo_name, (scores, steps) in results_dict.items():
        # Plot scores with moving average
        if len(scores) >= window:
            moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
            ax1.plot(range(window-1, len(scores)), moving_avg, 
                    linewidth=2, label=algo_name)
        
        # Plot steps with moving average
        if len(steps) >= window:
            moving_avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
            ax2.plot(range(window-1, len(steps)), moving_avg_steps,
                    linewidth=2, label=algo_name)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score (Moving Avg)')
    ax1.set_title('Algorithm Comparison - Scores')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps (Moving Avg)')
    ax2.set_title('Algorithm Comparison - Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{DIFFICULTY}.png', dpi=150)
    plt.show()
