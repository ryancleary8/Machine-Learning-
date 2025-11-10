# ============================================================================
# FILE: main.py
# ============================================================================
"""
Main entry point for Q-Learning Grid-World Maze Solver.
Ties together environment, agent, training, and visualization.
"""

from config import *
from environment import GridWorld
from agent_qlearning import QLearningAgent
from visualize import (plot_rewards, plot_steps, show_q_heatmap,
                        plot_maze_with_path, plot_policy,
                        create_comprehensive_visualization)


def main():
    """
    Main execution function.
    Sets up environment, trains agent, and generates visualizations.
    """
    print("=" * 70)
    print(" Q-LEARNING GRID-WORLD MAZE SOLVER")
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
    print("\n[2] Initializing Q-Learning Agent")
    print("-" * 70)
    
    agent = QLearningAgent(
        env=env,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN
    )
    
    print(f"    Learning Rate (α): {ALPHA}")
    print(f"    Discount Factor (γ): {GAMMA}")
    print(f"    Initial Exploration Rate (ε): {EPSILON}")
    print(f"    Epsilon Decay: {EPSILON_DECAY}")
    print(f"    Minimum Epsilon: {EPSILON_MIN}")
    print(f"    Q-Table Shape: {agent.q_table.shape}")
    
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
    plot_rewards(rewards_history, window=PLOT_WINDOW)
    plot_steps(steps_history, window=PLOT_WINDOW)
    show_q_heatmap(agent.q_table, env)
    plot_policy(agent.q_table, env)
    
    # Test path visualization
    plot_maze_with_path(env, test_path,
                       title='Test Episode (Greedy Policy)',
                       save_path='test_path.png')
    
    # Comprehensive visualization
    create_comprehensive_visualization(
        env, agent, rewards_history, steps_history, paths_history
    )
    
    print("\n    Generated Files:")
    print("      ✓ learning_curve.png")
    print("      ✓ steps_curve.png")
    print("      ✓ q_heatmap.png")
    print("      ✓ policy.png")
    print("      ✓ test_path.png")
    print("      ✓ comprehensive_results.png")
    
    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    print("\n[7] Summary")
    print("-" * 70)
    print(f"    ✓ Successfully trained Q-Learning agent for {EPISODES} episodes")
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

