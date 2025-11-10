# config.py - Central configuration for Snake AI Learning Simulator

# Algorithm Selection
ALGORITHM = "Q_LEARNING"  # Options: Q_LEARNING, SARSA, DQN

# Difficulty Settings
DIFFICULTY = "HARD"  # Options: EASY, MEDIUM, HARD

# Grid sizes based on difficulty
GRID_SIZE = {
    "EASY": 6,
    "MEDIUM": 10,
    "HARD": 14
}[DIFFICULTY]

# Game speed (frames per second)
GAME_SPEED = {
    "EASY": 5,
    "MEDIUM": 10,
    "HARD": 15
}[DIFFICULTY]

# Learning Parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Training Parameters
EPISODES = 5000
MAX_STEPS_PER_EPISODE = 1000

# Reward Structure
REWARD_FOOD = 10
REWARD_DEATH = -10
REWARD_STEP = -0.1

# DQN Specific Parameters
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10

# Visualization
CELL_SIZE = 40
SHOW_TRAINING = True
PLOT_INTERVAL = 10  # Update plot every N episodes

# Mode
MODE = "TRAIN"  # Options: TRAIN, PLAY
