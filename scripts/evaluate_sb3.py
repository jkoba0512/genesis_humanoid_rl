#!/usr/bin/env python3
"""
Evaluation script for trained SB3 models.
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def evaluate_model(model_path: str, n_episodes: int = 10, render: bool = False):
    """Evaluate a trained model."""

    print(f"=== Evaluating Model: {model_path} ===")

    # Check if model exists
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file {model_path}.zip not found")
        return

    # Create environment
    render_mode = "human" if render else None
    env = make_humanoid_env(
        episode_length=1000,
        simulation_fps=100,
        control_freq=20,
        target_velocity=1.0,
        render_mode=render_mode,
    )

    print(
        f"Environment created - Obs: {env.observation_space.shape}, Action: {env.action_space.shape}"
    )

    # Load model
    print("Loading model...")
    model = PPO.load(model_path)

    # Evaluate policy
    print(f"Evaluating for {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"Episode {episode + 1}/{n_episodes}...", end=" ")

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            # Optional: print progress for long episodes
            if episode_length % 200 == 0:
                print(".", end="")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f" Reward: {episode_reward:.2f}, Length: {episode_length}")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)

    print(f"\n=== Evaluation Results ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f} +/- {std_length:.1f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Worst episode reward: {min(episode_rewards):.2f}")

    # Performance assessment
    if mean_reward > 100:
        print("🎉 Excellent performance!")
    elif mean_reward > 50:
        print("✅ Good performance!")
    elif mean_reward > 0:
        print("🔄 Learning in progress...")
    else:
        print("🔴 Needs more training")

    env.close()
    return mean_reward, std_reward


def compare_models(model_paths: list, n_episodes: int = 5):
    """Compare multiple models."""

    print("=== Model Comparison ===")

    results = []
    for model_path in model_paths:
        if os.path.exists(model_path + ".zip"):
            mean_reward, std_reward = evaluate_model(
                model_path, n_episodes, render=False
            )
            results.append((model_path, mean_reward, std_reward))
        else:
            print(f"Skipping {model_path} - file not found")

    # Sort by mean reward
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n=== Comparison Results ===")
    for i, (model_path, mean_reward, std_reward) in enumerate(results):
        model_name = os.path.basename(model_path)
        print(f"{i+1}. {model_name}: {mean_reward:.2f} +/- {std_reward:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate trained SB3 humanoid models")
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the trained model (without .zip extension)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate (default: 10)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render episodes during evaluation"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Compare multiple models (provide multiple model paths)",
    )

    args = parser.parse_args()

    try:
        if args.compare:
            compare_models(args.compare, args.episodes)
        else:
            evaluate_model(args.model_path, args.episodes, args.render)
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
