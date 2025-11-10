# main.py - Main entry point for Snake AI Learning Simulator

import sys
import os
from config import *
from environment import SnakeEnvironment
from visualize import SnakeVisualizer, plot_training_progress

# Import agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent

def load_agent(algorithm, state_size, action_size):
    """Load the selected algorithm agent"""
    if algorithm == "Q_LEARNING":
        return QLearningAgent(state_size, action_size)
    elif algorithm == "SARSA":
        return SARSAAgent(state_size, action_size)
    elif algorithm == "DQN":
        return DQNAgent(state_size, action_size)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def train_mode():
    """Training mode - train the agent"""
    print("="*60)
    print(f"🐍 Snake AI Learning Simulator - Training Mode")
    print("="*60)
    print(f"Algorithm: {ALGORITHM}")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Episodes: {EPISODES}")
    print("="*60)
    
    # Initialize environment
    env = SnakeEnvironment(GRID_SIZE)
    state_size = len(env.get_state())
    action_size = 3  # straight, left, right
    
    # Load agent
    agent = load_agent(ALGORITHM, state_size, action_size)
    
    # Optional: Initialize visualizer
    visualizer = None
    if SHOW_TRAINING:
        visualizer = SnakeVisualizer(GRID_SIZE)
    
    # Train the agent
    print("\nStarting training...\n")
    
    try:
        if visualizer:
            # Training with visualization (slower)
            for episode in range(EPISODES):
                state = env.reset()
                total_reward = 0
                steps = 0
                
                # For SARSA, choose initial action
                if ALGORITHM == "SARSA":
                    action = agent.choose_action(state)
                
                while not env.done and steps < MAX_STEPS_PER_EPISODE:
                    # Render every N episodes
                    if episode % 1000 == 0:
                        if not visualizer.render(env, env.score, episode):
                            raise KeyboardInterrupt
                    
                    if ALGORITHM == "SARSA":
                        next_state, reward, done = env.step(action)
                        next_action = agent.choose_action(next_state)
                        agent.update(state, action, reward, next_state, next_action, done)
                        action = next_action
                    else:
                        action = agent.choose_action(state)
                        next_state, reward, done = env.step(action)
                        agent.update(state, action, reward, next_state, done)
                    
                    state = next_state
                    total_reward += reward
                    steps += 1
                
                # Decay epsilon
                agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
                
                agent.training_scores.append(env.score)
                agent.training_steps.append(steps)
                
                if (episode + 1) % PLOT_INTERVAL == 0:
                    avg_score = np.mean(agent.training_scores[-PLOT_INTERVAL:])
                    print(f"Episode {episode + 1}/{EPISODES}, "
                          f"Avg Score: {avg_score:.2f}, "
                          f"Score: {env.score}, "
                          f"Steps: {steps}, "
                          f"Epsilon: {agent.epsilon:.3f}")
            
            visualizer.close()
            scores, steps = agent.training_scores, agent.training_steps
        else:
            # Training without visualization (faster)
            scores, steps = agent.train(env, EPISODES)
        
        # Save the trained agent
        save_path = f"saved_models/{ALGORITHM}_{DIFFICULTY}.pkl"
        os.makedirs("saved_models", exist_ok=True)
        agent.save(save_path)
        
        # Plot training progress
        print("\nTraining completed! Generating plots...")
        plot_training_progress(scores, steps, f"{ALGORITHM}_{DIFFICULTY}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Final average score (last 50 episodes): {np.mean(scores[-50:]):.2f}")
        print(f"💾 Model saved to: {save_path}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        if visualizer:
            visualizer.close()

def play_mode():
    """Play mode - watch the trained agent play"""
    print("="*60)
    print(f"🐍 Snake AI Learning Simulator - Play Mode")
    print("="*60)
    print(f"Algorithm: {ALGORITHM}")
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print("="*60)
    
    # Initialize environment
    env = SnakeEnvironment(GRID_SIZE)
    state_size = len(env.get_state())
    action_size = 3
    
    # Load agent
    agent = load_agent(ALGORITHM, state_size, action_size)
    
    # Load trained model
    load_path = f"saved_models/{ALGORITHM}_{DIFFICULTY}.pkl"
    try:
        agent.load(load_path)
        agent.epsilon = 0  # No exploration during play
    except FileNotFoundError:
        print(f"\n❌ No trained model found at {load_path}")
        print("Please train the agent first using MODE='TRAIN' in config.py")
        return
    
    # Initialize visualizer
    visualizer = SnakeVisualizer(GRID_SIZE)
    
    print("\nWatching trained agent play...")
    print("Press ESC or close window to exit\n")
    
    try:
        games_played = 0
        total_scores = []
        
        while True:
            state = env.reset()
            
            while not env.done:
                if not visualizer.render(env, env.score, f"Game {games_played + 1}"):
                    raise KeyboardInterrupt
                
                action = agent.choose_action(state)
                state, reward, done = env.step(action)
            
            games_played += 1
            total_scores.append(env.score)
            
            print(f"Game {games_played}: Score = {env.score}, "
                  f"Average = {np.mean(total_scores):.2f}")
            
            # Wait a bit before next game
            import time
            time.sleep(1)
    
    except KeyboardInterrupt:
        print(f"\n\n📊 Statistics:")
        print(f"Games played: {games_played}")
        print(f"Average score: {np.mean(total_scores):.2f}")
        print(f"Best score: {max(total_scores)}")
        visualizer.close()

if __name__ == "__main__":
    import numpy as np
    
    if MODE == "TRAIN":
        train_mode()
    elif MODE == "PLAY":
        play_mode()
    else:
        print(f"Invalid MODE in config.py: {MODE}")
        print("Please set MODE to either 'TRAIN' or 'PLAY'")
