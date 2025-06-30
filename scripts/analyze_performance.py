#!/usr/bin/env python3
"""
Analyze performance of the trained model.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def main():
    parser = argparse.ArgumentParser(description="Analyze trained model performance")
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to analyze"
    )
    parser.add_argument(
        "--save-plots", action="store_true", help="Save performance plots"
    )
    args = parser.parse_args()

    print("=== Performance Analysis ===")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")

    # Load model
    model = PPO.load(args.model)

    # Create environment (no rendering for faster analysis)
    env = make_humanoid_env(
        episode_length=200, simulation_fps=50, control_freq=20, target_velocity=1.0
    )

    # Collect performance data
    episode_rewards = []
    episode_lengths = []
    all_rewards = []

    print("\nRunning analysis...")
    for episode in range(args.episodes):
        obs, _ = env.reset()
        episode_reward = 0
        step_count = 0
        episode_reward_history = []

        for step in range(200):  # Max 200 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_count += 1
            episode_reward_history.append(reward)

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        all_rewards.extend(episode_reward_history)

        print(
            f"Episode {episode+1}: {step_count} steps, {episode_reward:.2f} total reward"
        )

    # Calculate statistics
    print(f"\n=== Results ===")
    print(f"Episodes completed: {len(episode_rewards)}")
    print(
        f"Average episode reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
    )
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Worst episode reward: {np.min(episode_rewards):.2f}")
    print(f"Average step reward: {np.mean(all_rewards):.3f}")

    # Create plots if requested
    if args.save_plots:
        plt.figure(figsize=(12, 8))

        # Episode rewards
        plt.subplot(2, 2, 1)
        plt.plot(episode_rewards, "b-o")
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)

        # Episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(episode_lengths, "r-o")
        plt.title("Episode Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)

        # Reward distribution
        plt.subplot(2, 2, 3)
        plt.hist(all_rewards, bins=30, alpha=0.7)
        plt.title("Step Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Moving average of rewards
        plt.subplot(2, 2, 4)
        window = min(50, len(all_rewards) // 4)
        moving_avg = np.convolve(all_rewards, np.ones(window) / window, mode="valid")
        plt.plot(moving_avg)
        plt.title(f"Moving Average Reward (window={window})")
        plt.xlabel("Step")
        plt.ylabel("Average Reward")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("model_performance_analysis.png", dpi=150, bbox_inches="tight")
        print(f"\n📊 Performance plots saved to: model_performance_analysis.png")

    env.close()
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
