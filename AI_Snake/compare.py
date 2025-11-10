# compare.py - Compare all algorithms side-by-side

import sys
import os
import numpy as np
from environment import SnakeEnvironment
from visualize import compare_algorithms
from config import GRID_SIZE, EPISODES

# Import agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from agents.qlearning import QLearningAgent
from agents.sarsa import SARSAAgent
from agents.dqn import DQNAgent

def compare_all_algorithms(episodes=200):
    """Train and compare all algorithms"""
    print("="*60)
    print("🐍 Snake AI - Algorithm Comparison")
    print("="*60)
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Episodes per algorithm: {episodes}")
    print("="*60)
    
    # Initialize environment
    env = SnakeEnvironment(GRID_SIZE)
    state_size = len(env.get_state())
    action_size = 3
    
    results = {}
    
    # List of algorithms to compare
    algorithms = [
        ("Q-Learning", QLearningAgent),
        ("SARSA", SARSAAgent),
        # Uncomment if you have PyTorch installed
        # ("DQN", DQNAgent),
    ]
    
    for algo_name, AgentClass in algorithms:
        print(f"\n{'='*60}")
        print(f"Training {algo_name}...")
        print('='*60)
        
        agent = AgentClass(state_size, action_size)
        scores, steps = agent.train(env, episodes)
        results[algo_name] = (scores, steps)
        
        # Print summary statistics
        print(f"\n{algo_name} Results:")
        print(f"  Average score (all): {np.mean(scores):.2f}")
        print(f"  Average score (last 50): {np.mean(scores[-50:]):.2f}")
        print(f"  Best score: {max(scores)}")
        print(f"  Average steps: {np.mean(steps):.2f}")
    
    # Plot comparison
    print("\n" + "="*60)
    print("Generating comparison plots...")
    print("="*60)
    compare_algorithms(results)
    
    # Print final comparison
    print("\n" + "="*60)
    print("📊 Final Comparison (Last 50 Episodes Average):")
    print("="*60)
    for algo_name, (scores, _) in results.items():
        avg_score = np.mean(scores[-50:])
        print(f"{algo_name:15s}: {avg_score:6.2f}")
    print("="*60)

if __name__ == "__main__":
    # You can adjust the number of episodes here
    compare_all_algorithms(episodes=EPISODES)
