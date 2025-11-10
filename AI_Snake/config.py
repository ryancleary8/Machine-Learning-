# config.py - Central configuration for Snake AI Learning Simulator

# Algorithm Selection
ALGORITHM = "SARSA"  # Options: Q_LEARNING, SARSA, DQN, A_STAR, PPO

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
REWARD_DEATH = -50
REWARD_SELF_COLLISION = -75
REWARD_STEP = -0.1
REWARD_STRAIGHT = 0.2

# DQN Specific Parameters
BATCH_SIZE = 64
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 10

# PPO Specific Parameters
PPO_CLIP_EPS = 0.2
PPO_EPOCHS = 4
PPO_BATCH_SIZE = 64
PPO_GAE_LAMBDA = 0.95
PPO_ENTROPY_COEF = 0.01

# Visualization
CELL_SIZE = 40
SHOW_TRAINING = True
PLOT_INTERVAL = 10  # Update plot every N episodes

# Mode
MODE = "TRAIN"  # Options: TRAIN, PLAY
