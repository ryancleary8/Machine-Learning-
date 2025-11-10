# ============================================================================
# FILE: main.py
# ============================================================================
"""
Main entry point for Q-Learning Grid-World Maze Solver.
Ties together environment, agent, training, and visualization.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from config import *
from environment import GridWorld
from agent_qlearning import QLearningAgent
from agent_sarsa import SARSAAgent
from agent_expected_sarsa import ExpectedSARSAAgent
from agent_double_q import DoubleQLearningAgent
from agent_dqn import DQNAgent, DoubleDQNAgent
from visualize import (plot_rewards, plot_steps, show_q_heatmap,
                        plot_maze_with_path, plot_policy,
                        create_comprehensive_visualization)


AGENT_REGISTRY = {
        "q_learning": {"label": "Q-Learning", "cls": QLearningAgent, "family": "tabular"},
    "sarsa": {"label": "SARSA", "cls": SARSAAgent, "family": "tabular"},
    "expected_sarsa": {"label": "Expected SARSA", "cls": ExpectedSARSAAgent, "family": "tabular"},
    "double_q": {"label": "Double Q-Learning", "cls": DoubleQLearningAgent, "family": "tabular"},
    "dqn": {"label": "Deep Q-Network", "cls": DQNAgent, "family": "dqn"},
    "double_dqn": {"label": "Double DQN", "cls": DoubleDQNAgent, "family": "dqn"},
}


def main():
    """
    Main execution function.
    Sets up environment, trains agent, and generates visualizations.
    """
    algorithm_key = AGENT_ALGORITHM.lower()
    if algorithm_key not in AGENT_REGISTRY:
        raise ValueError(f"Unsupported agent algorithm: '{AGENT_ALGORITHM}'")

    registry_entry = AGENT_REGISTRY[algorithm_key]
    algorithm_label = registry_entry["label"]
    agent_cls = registry_entry["cls"]
    agent_family = registry_entry["family"]

    print("=" * 70)
    print(f" {algorithm_label.upper()} GRID-WORLD MAZE SOLVER")
    print("=" * 70)
    
    # ========================================================================
    # STEP 1: Create Environment
    # ========================================================================
    print("\n[1] Creating Grid-World Environment")
    print("-" * 70)
    
    env = GridWorld(
        grid_size=GRID_SIZE,
        start_pos=START_POS,
        goal_pos=GOAL_POS,
        obstacles=OBSTACLES,
        reward_goal=REWARD_GOAL,
        reward_wall=REWARD_WALL,
        reward_step=REWARD_STEP
    )
    
    print(f"    Grid Size: {GRID_SIZE[0]}x{GRID_SIZE[1]}")
    print(f"    Start Position: {START_POS}")
    print(f"    Goal Position: {GOAL_POS}")
    print(f"    Number of Obstacles: {len(OBSTACLES)}")
    print(f"    Total States: {env.rows * env.cols}")
    print(f"    Actions: {len(env.actions)} (Up, Down, Left, Right)")
    print(f"\n    Reward Structure:")
    print(f"      - Goal: {REWARD_GOAL}")
    print(f"      - Wall/Boundary: {REWARD_WALL}")
    print(f"      - Step: {REWARD_STEP}")
    
    # Print ASCII representation of maze
    print(f"\n    Initial Maze Layout:")
    print(env.render())
    
    # ========================================================================
    # STEP 2: Initialize Q-Learning Agent
    # ========================================================================
    print("\n[2] Initializing Agent")
    print("-" * 70)

    print(f"    Selected Algorithm: {algorithm_label}")

    if agent_family == "tabular":
        agent = agent_cls(
            env=env,
            alpha=ALPHA,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN
        )
    elif agent_family == "dqn":
        agent = agent_cls(
            env=env,
            alpha=DQN_LEARNING_RATE,
            gamma=GAMMA,
            epsilon=EPSILON,
            epsilon_decay=EPSILON_DECAY,
            epsilon_min=EPSILON_MIN,
            batch_size=DQN_BATCH_SIZE,
            buffer_size=DQN_BUFFER_SIZE,
            target_update=DQN_TARGET_UPDATE,
            hidden_units=DQN_HIDDEN_UNITS,
        )
    else:
        raise ValueError(f"Unknown agent family '{agent_family}' for algorithm '{algorithm_key}'")
        
        
    
    if agent_family == "tabular":
        print(f"    Learning Rate (α): {ALPHA}")
    else:
        print(f"    Learning Rate (α): {DQN_LEARNING_RATE}")
        
    print(f"    Discount Factor (γ): {GAMMA}")
    print(f"    Initial Exploration Rate (ε): {EPSILON}")
    print(f"    Epsilon Decay: {EPSILON_DECAY}")
    print(f"    Minimum Epsilon: {EPSILON_MIN}")
    if hasattr(agent, "q_table"):
        print(f"    Q-Table Shape: {agent.q_table.shape}")
    if agent_family == "dqn":
        print(f"    Neural Network Hidden Units: {DQN_HIDDEN_UNITS}")
        print(f"    Replay Buffer Size: {DQN_BUFFER_SIZE}")
        print(f"    Batch Size: {DQN_BATCH_SIZE}")
        print(f"    Target Update Frequency: {DQN_TARGET_UPDATE} episodes")

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"    Results Directory: {os.path.abspath(RESULTS_DIR)}")
    
    # ========================================================================
    # STEP 3: Train Agent
    # ========================================================================
    print("\n[3] Training Agent")
    print("-" * 70)
    
    rewards_history, steps_history, paths_history = agent.train(
        num_episodes=EPISODES,
        max_steps=MAX_STEPS,
        episodes_to_save=SAVE_PATHS_AT
    )
    
    # ========================================================================
    # STEP 4: Display Training Statistics
    # ========================================================================
    print("\n[4] Training Statistics")
    print("-" * 70)
    
    # Calculate statistics
    initial_avg_reward = np.mean(rewards_history[:100])
    final_avg_reward = np.mean(rewards_history[-100:])
    initial_avg_steps = np.mean(steps_history[:100])
    final_avg_steps = np.mean(steps_history[-100:])
    
    print(f"    Initial Performance (first 100 episodes):")
    print(f"      - Average Reward: {initial_avg_reward:.2f}")
    print(f"      - Average Steps: {initial_avg_steps:.2f}")
    print(f"\n    Final Performance (last 100 episodes):")
    print(f"      - Average Reward: {final_avg_reward:.2f}")
    print(f"      - Average Steps: {final_avg_steps:.2f}")
    print(f"\n    Improvement:")
    print(f"      - Reward Gain: {final_avg_reward - initial_avg_reward:+.2f}")
    print(f"      - Steps Reduced: {initial_avg_steps - final_avg_steps:.2f}")
    print(f"\n    Final Epsilon: {agent.epsilon:.4f}")
    
    # ========================================================================
    # STEP 5: Test Learned Policy
    # ========================================================================
    print("\n[5] Testing Learned Policy (Greedy)")
    print("-" * 70)
    
    # Test with pure exploitation (no exploration)
    agent.epsilon = 0.0
    state = env.reset()
    test_path = [env.current_state]
    total_test_reward = 0
    
    for step in range(MAX_STEPS):
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        test_path.append(env.current_state)
        total_test_reward += reward
        if done:
            break
    
    print(f"    Test Episode Results:")
    print(f"      - Steps Taken: {len(test_path) - 1}")
    print(f"      - Total Reward: {total_test_reward}")
    print(f"      - Goal Reached: {env.current_state == env.goal}")
    print(f"      - Optimal Path Length: ~{abs(GOAL_POS[0] - START_POS[0]) + abs(GOAL_POS[1] - START_POS[1])}")
    
    # ========================================================================
    # STEP 6: Generate Visualizations
    # ========================================================================
    print("\n[6] Generating Visualizations")
    print("-" * 70)
    
    # Individual plots
    rewards_path = os.path.join(RESULTS_DIR, 'learning_curve.png')
    steps_path = os.path.join(RESULTS_DIR, 'steps_curve.png')
    heatmap_path = os.path.join(RESULTS_DIR, 'q_heatmap.png')
    policy_path = os.path.join(RESULTS_DIR, 'policy.png')
    test_path_file = os.path.join(RESULTS_DIR, 'test_path.png')
    comprehensive_path = os.path.join(
        RESULTS_DIR, f"{algorithm_key}_results.png"
    )

    plot_rewards(rewards_history, window=PLOT_WINDOW, save_path=rewards_path)
    plot_steps(steps_history, window=PLOT_WINDOW, save_path=steps_path)
    show_q_heatmap(agent.q_table, env, save_path=heatmap_path)
    plot_policy(agent.q_table, env, save_path=policy_path)
    
    # Test path visualization
    plot_maze_with_path(
        env,
        test_path,
        title='Test Episode (Greedy Policy)',
        save_path=test_path_file,
    )

    # Comprehensive visualization
    create_comprehensive_visualization(
        env,
        agent,
        rewards_history,
        steps_history,
        paths_history,
        save_path=comprehensive_path,
    )

    print("\n    Generated Files:")
    generated_files = [
        rewards_path,
        steps_path,
        heatmap_path,
        policy_path,
        test_path_file,
        comprehensive_path,
    ]
    for file_path in generated_files:
        print(f"      ✓ {os.path.relpath(file_path, RESULTS_DIR)}")
    
    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    print("\n[7] Summary")
    print("-" * 70)
    print(f"    ✓ Successfully trained {algorithm_label} agent for {EPISODES} episodes")
    print(f"    ✓ Agent learned to navigate from {START_POS} to {GOAL_POS}")
    print(f"    ✓ Final performance: {final_avg_steps:.1f} steps (vs {initial_avg_steps:.1f} initially)")
    print(f"    ✓ All visualizations saved successfully")
    
    print("\n" + "=" * 70)
    print(" TRAINING COMPLETE!")
    print("=" * 70)
    print("\n Check the generated PNG files for detailed visualizations.\n")
    
    return env, agent, rewards_history, steps_history, paths_history


if __name__ == "__main__":
    # Execute main training pipeline
    env, agent, rewards, steps, paths = main()
    
    # Optional: Display final plot
    plt.show()

