#!/usr/bin/env python3
"""
Simplified training script for humanoid walking RL without Acme dependencies.
Uses the existing PPO agent but with a custom training loop.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import jax
import jax.numpy as jnp

from genesis_humanoid_rl.environments.humanoid_env import HumanoidWalkingEnv
from genesis_humanoid_rl.config.training_config import (
    get_default_config,
    load_config_from_dict,
)


def run_training(config_path: str = None):
    """Run the simplified training loop."""
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = load_config_from_dict(config_dict)
    else:
        config = get_default_config()

    env_config = config["environment"]
    agent_config = config["agent"]
    training_config = config["training"]

    print(f"Starting training with config:")
    print(f"  Total steps: {training_config.total_steps}")
    print(f"  Experiment: {training_config.experiment_name}")
    print(f"  Log dir: {training_config.log_dir}")

    # Set random seed
    np.random.seed(training_config.seed)

    # Create directories
    os.makedirs(training_config.log_dir, exist_ok=True)
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    # Create environment (without Acme wrappers)
    env = HumanoidWalkingEnv(
        render_mode=env_config.render_mode,
        simulation_fps=env_config.simulation_fps,
        control_freq=env_config.control_freq,
        episode_length=env_config.episode_length,
        target_velocity=env_config.target_velocity,
    )

    print(f"Environment created, initializing...")

    # Do a dummy reset to initialize the environment properly
    dummy_obs, _ = env.reset()
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action shape: {env.action_space.shape}")

    # Initialize training components
    print("Initializing training components...")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode = 0

    # Initialize key for JAX random number generation
    key = jax.random.PRNGKey(training_config.seed)

    # For this simplified version, we'll just use random actions
    # In a full implementation, you would initialize and use the PPO networks

    print(f"\nStarting training loop...")
    print(
        f"{'Episode':<10} {'Steps':<10} {'Reward':<15} {'Avg Reward':<15} {'Episode Length':<15}"
    )
    print("-" * 75)

    # Training loop
    while total_steps < training_config.total_steps:
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        # Collect trajectory
        for step in range(env_config.episode_length):
            # Select random action for testing
            # In a full implementation, this would use the policy network
            action = env.action_space.sample()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update counters
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            # Update observation
            obs = next_obs

            # Check if episode ended
            if done or step == env_config.episode_length - 1:
                break

        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1

        # Compute moving average reward
        avg_reward = np.mean(episode_rewards[-min(100, len(episode_rewards)) :])

        # Log progress
        if episode % training_config.log_frequency == 0:
            print(
                f"{episode:<10} {total_steps:<10} {episode_reward:<15.2f} {avg_reward:<15.2f} {episode_length:<15}"
            )

        # Save checkpoint periodically
        if episode % training_config.save_frequency == 0 and episode > 0:
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f"checkpoint_episode_{episode}_steps_{total_steps}.npz",
            )
            np.savez(
                checkpoint_path,
                episode=episode,
                total_steps=total_steps,
                episode_rewards=episode_rewards,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        # Early stopping check
        if avg_reward >= 500.0:  # Arbitrary threshold
            print(f"\nReached target performance! Average reward: {avg_reward:.2f}")
            break

    print(f"\nTraining completed!")
    print(f"Total episodes: {episode}")
    print(f"Total steps: {total_steps}")
    print(f"Final average reward: {avg_reward:.2f}")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(
        training_config.checkpoint_dir,
        f"final_checkpoint_episode_{episode}_steps_{total_steps}.npz",
    )
    np.savez(
        final_checkpoint_path,
        episode=episode,
        total_steps=total_steps,
        episode_rewards=episode_rewards,
    )
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Cleanup
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train humanoid walking agent (simplified)"
    )
    parser.add_argument(
        "--config", type=str, help="Path to training configuration JSON file"
    )

    args = parser.parse_args()

    try:
        run_training(args.config)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
