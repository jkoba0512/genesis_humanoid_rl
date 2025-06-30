#!/usr/bin/env python3
"""
Training script for humanoid walking RL.
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
from acme import specs, wrappers
from acme.utils import loggers

from genesis_humanoid_rl.environments.humanoid_env import HumanoidWalkingEnv
from genesis_humanoid_rl.agents.ppo_agent import PPOHumanoidAgent
from genesis_humanoid_rl.config.training_config import (
    get_default_config,
    load_config_from_dict,
)


def create_environment(env_config) -> HumanoidWalkingEnv:
    """Create the humanoid walking environment."""
    env = HumanoidWalkingEnv(
        render_mode=env_config.render_mode,
        simulation_fps=env_config.simulation_fps,
        control_freq=env_config.control_freq,
        episode_length=env_config.episode_length,
        target_velocity=env_config.target_velocity,
    )

    # Wrap environment for Acme compatibility
    env = wrappers.GymWrapper(env)
    env = wrappers.SinglePrecisionWrapper(env)

    return env


def create_agent(
    environment_spec: specs.EnvironmentSpec, agent_config
) -> PPOHumanoidAgent:
    """Create the PPO agent."""
    network_kwargs = {
        "policy_layers": agent_config.policy_layers,
        "value_layers": agent_config.value_layers,
        "activation": getattr(jax.nn, agent_config.activation),
    }

    agent = PPOHumanoidAgent(
        environment_spec=environment_spec,
        network_kwargs=network_kwargs,
        learning_rate=agent_config.learning_rate,
        entropy_cost=agent_config.entropy_cost,
        value_cost=agent_config.value_cost,
        max_gradient_norm=agent_config.max_gradient_norm,
        num_epochs=agent_config.num_epochs,
        num_minibatches=agent_config.num_minibatches,
        unroll_length=agent_config.unroll_length,
        batch_size=agent_config.batch_size,
    )

    return agent


def run_training(config_path: str = None, resume_from: str = None):
    """Run the training loop."""
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

    # Create environment
    env = create_environment(env_config)
    environment_spec = specs.make_environment_spec(env)

    print(f"Environment created:")
    print(f"  Observation shape: {environment_spec.observations.shape}")
    print(f"  Action shape: {environment_spec.actions.shape}")

    # Create agent
    agent = create_agent(environment_spec, agent_config)

    print("Agent created, building networks...")
    agent.build_agent()
    print("Agent ready for training!")

    # Initialize training components
    print("Initializing training components...")

    # Training metrics
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    episode = 0

    # Initialize key for JAX random number generation
    key = jax.random.PRNGKey(training_config.seed)

    # Create a dummy observation to initialize networks
    dummy_obs = np.zeros(environment_spec.observations.shape, dtype=np.float32)
    dummy_obs_batch = np.expand_dims(dummy_obs, axis=0)

    # Get networks from agent
    networks = agent._networks

    # Initialize network parameters
    key, init_key = jax.random.split(key)
    params = {
        "policy": networks.policy_network.init(init_key, dummy_obs_batch),
        "value": networks.value_network.init(init_key, dummy_obs_batch),
    }

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
            # Select action using policy
            key, subkey = jax.random.split(key)
            obs_batch = np.expand_dims(obs, axis=0)  # Add batch dimension

            # Sample action from policy using the network's sample function
            action = networks.sample(params["policy"], obs_batch, subkey)
            action = np.squeeze(action, axis=0)  # Remove batch dimension

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
        if episode % training_config.log_interval == 0:
            print(
                f"{episode:<10} {total_steps:<10} {episode_reward:<15.2f} {avg_reward:<15.2f} {episode_length:<15}"
            )

        # Save checkpoint periodically
        if episode % training_config.checkpoint_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(
                training_config.checkpoint_dir,
                f"checkpoint_episode_{episode}_steps_{total_steps}.npz",
            )
            np.savez(
                checkpoint_path,
                params=params,
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
        params=params,
        episode=episode,
        total_steps=total_steps,
        episode_rewards=episode_rewards,
    )
    print(f"Saved final checkpoint to {final_checkpoint_path}")

    # Cleanup
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train humanoid walking agent")
    parser.add_argument(
        "--config", type=str, help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--resume-from", type=str, help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment during training"
    )

    args = parser.parse_args()

    # Override render mode if specified
    if args.render:
        print("Rendering enabled")

    try:
        run_training(args.config, args.resume_from)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
