#!/usr/bin/env python3
"""
Training script using Stable-Baselines3 for humanoid walking.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


class TensorboardCallback:
    """Custom callback for additional TensorBoard logging."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []

    def __call__(
        self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]
    ) -> bool:
        # This callback is called after each environment step
        # We can add custom logging here if needed
        return True


def create_training_config():
    """Create default training configuration for SB3."""
    return {
        "env": {
            "episode_length": 1000,
            "simulation_fps": 100,
            "control_freq": 20,
            "target_velocity": 1.0,
            "n_envs": 4,  # Number of parallel environments
        },
        "algorithm": {
            "learning_rate": 3e-4,
            "n_steps": 2048,  # Steps per environment per update
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256, 128],  # Shared network architecture
                "activation_fn": "tanh",
            },
        },
        "training": {
            "total_timesteps": 1_000_000,
            "save_freq": 50_000,
            "eval_freq": 10_000,
            "eval_episodes": 5,
            "log_interval": 10,
            "experiment_name": "humanoid_walk_sb3",
            "log_dir": "./logs/sb3",
            "model_dir": "./models/sb3",
        },
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            user_config = json.load(f)

        # Merge with default config
        config = create_training_config()

        # Update nested dictionaries
        for section, values in user_config.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values

        return config
    else:
        # Try to load default config from configs directory
        default_config_path = "configs/default.json"
        if os.path.exists(default_config_path):
            with open(default_config_path, "r") as f:
                return json.load(f)
        else:
            return create_training_config()


def create_environment(env_config: Dict[str, Any], render_mode: str = None):
    """Create the training environment."""

    def _make_env():
        return make_humanoid_env(
            episode_length=env_config["episode_length"],
            simulation_fps=env_config["simulation_fps"],
            control_freq=env_config["control_freq"],
            target_velocity=env_config["target_velocity"],
            render_mode=render_mode,
        )

    # Create vectorized environment
    if env_config["n_envs"] > 1:
        env = make_vec_env(_make_env, n_envs=env_config["n_envs"], seed=42)
        print(f"Created {env_config['n_envs']} parallel environments")
    else:
        env = Monitor(_make_env())
        print("Created single environment")

    return env


def train(config_path: str = None, render: bool = False):
    """Main training function."""

    # Load configuration
    config = load_config(config_path)
    env_config = config["env"]
    algo_config = config["algorithm"]
    training_config = config["training"]

    print("=== Stable-Baselines3 Humanoid Training ===")
    print(f"Experiment: {training_config['experiment_name']}")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Parallel environments: {env_config['n_envs']}")
    print(f"Episode length: {env_config['episode_length']}")

    # Create directories
    os.makedirs(training_config["log_dir"], exist_ok=True)
    os.makedirs(training_config["model_dir"], exist_ok=True)

    # Create environment
    render_mode = "human" if render else None
    env = create_environment(env_config, render_mode=render_mode)

    # Create evaluation environment (single environment for consistent evaluation)
    # Skip eval environment for quick tests to avoid double kernel compilation
    eval_env = None
    if (
        training_config.get("eval_freq", 0) > 0
        and training_config["total_timesteps"] > 10000
    ):
        eval_env = Monitor(
            make_humanoid_env(
                episode_length=env_config["episode_length"],
                simulation_fps=env_config["simulation_fps"],
                control_freq=env_config["control_freq"],
                target_velocity=env_config["target_velocity"],
            )
        )

    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")

    # Configure logger
    logger = configure(training_config["log_dir"], ["stdout", "tensorboard"])

    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=algo_config["learning_rate"],
        n_steps=algo_config["n_steps"],
        batch_size=algo_config["batch_size"],
        n_epochs=algo_config["n_epochs"],
        gamma=algo_config["gamma"],
        gae_lambda=algo_config["gae_lambda"],
        clip_range=algo_config["clip_range"],
        ent_coef=algo_config["ent_coef"],
        vf_coef=algo_config["vf_coef"],
        max_grad_norm=algo_config["max_grad_norm"],
        policy_kwargs=algo_config["policy_kwargs"],
        verbose=1,
        seed=42,
        device="auto",  # Use GPU if available
    )

    # Set custom logger
    model.set_logger(logger)

    # Create callbacks
    callbacks = []

    # Evaluation callback (only if eval environment was created)
    if eval_env is not None:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(
                training_config["model_dir"], "best_model"
            ),
            log_path=training_config["log_dir"],
            eval_freq=training_config["eval_freq"]
            // env_config["n_envs"],  # Adjust for vectorized env
            n_eval_episodes=training_config["eval_episodes"],
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config["save_freq"]
        // env_config["n_envs"],  # Adjust for vectorized env
        save_path=os.path.join(training_config["model_dir"], "checkpoints"),
        name_prefix="ppo_humanoid",
    )
    callbacks.append(checkpoint_callback)

    callback_list = CallbackList(callbacks)

    print(f"\nStarting training...")
    print(f"Logs will be saved to: {training_config['log_dir']}")
    print(f"Models will be saved to: {training_config['model_dir']}")
    print(f"Monitor training with: tensorboard --logdir {training_config['log_dir']}")

    try:
        # Train the model
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=callback_list,
            log_interval=training_config["log_interval"],
            progress_bar=True,
        )

        # Save final model
        final_model_path = os.path.join(training_config["model_dir"], "final_model")
        model.save(final_model_path)
        print(f"\n✓ Training completed! Final model saved to: {final_model_path}")

        # Test the trained model (only if eval environment exists)
        if eval_env is not None:
            print("\nTesting trained model...")
            test_episodes = 3
            for episode in range(test_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0
                done = False

                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated

                print(f"Test episode {episode + 1}: Reward = {episode_reward:.2f}")
        else:
            print("\nSkipping model test (no evaluation environment created)")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save current model
        interrupt_model_path = os.path.join(
            training_config["model_dir"], "interrupted_model"
        )
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")

    finally:
        env.close()
        if eval_env is not None:
            eval_env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train humanoid walking with Stable-Baselines3"
    )
    parser.add_argument(
        "--config", type=str, help="Path to training configuration JSON file"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during training (slower)",
    )

    args = parser.parse_args()

    try:
        train(args.config, args.render)
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
