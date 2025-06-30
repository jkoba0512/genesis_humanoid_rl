#!/usr/bin/env python3
"""
Training script with curriculum learning for humanoid walking.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any
import time
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from genesis_humanoid_rl.environments.curriculum_env import CurriculumHumanoidEnv
from genesis_humanoid_rl.curriculum.curriculum_manager import CurriculumStage


class CurriculumCallback(BaseCallback):
    """Custom callback to monitor curriculum learning progress."""

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.stage_changes = []

    def _on_step(self) -> bool:
        # Get info from the environment
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                # Episode finished
                episode_reward = info["episode"]["r"]
                episode_length = info["episode"]["l"]

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)

                # Log curriculum info if available
                if "curriculum" in info:
                    curr_info = info["curriculum"]

                    # Log to tensorboard
                    self.logger.record(
                        "curriculum/stage", curr_info.get("current_stage", "unknown")
                    )
                    self.logger.record(
                        "curriculum/target_velocity",
                        curr_info.get("target_velocity", 0.0),
                    )
                    self.logger.record(
                        "curriculum/recent_avg_reward",
                        curr_info.get("recent_avg_reward", 0.0),
                    )
                    self.logger.record(
                        "curriculum/total_episodes", curr_info.get("total_episodes", 0)
                    )

                    # Print stage changes
                    current_stage = curr_info.get("current_stage", "unknown")
                    if (
                        not self.stage_changes
                        or self.stage_changes[-1] != current_stage
                    ):
                        self.stage_changes.append(current_stage)
                        print(f"\n🎓 CURRICULUM STAGE: {current_stage}")
                        print(f"   Episode Reward: {episode_reward:.2f}")
                        print(f"   Episode Length: {episode_length}")
                        print(
                            f"   Target Velocity: {curr_info.get('target_velocity', 0.0):.2f}"
                        )

        # Log periodic statistics
        if (
            len(self.episode_rewards) > 0
            and len(self.episode_rewards) % self.log_freq == 0
        ):
            recent_rewards = self.episode_rewards[-self.log_freq :]
            recent_lengths = self.episode_lengths[-self.log_freq :]

            self.logger.record("rollout/ep_rew_mean_recent", np.mean(recent_rewards))
            self.logger.record("rollout/ep_len_mean_recent", np.mean(recent_lengths))

            print(f"\n📊 Last {self.log_freq} episodes:")
            print(f"   Mean Reward: {np.mean(recent_rewards):.3f}")
            print(f"   Mean Length: {np.mean(recent_lengths):.1f}")
            print(
                f"   Current Stage: {self.stage_changes[-1] if self.stage_changes else 'unknown'}"
            )

        return True


def create_curriculum_config():
    """Create default curriculum training configuration."""
    return {
        "env": {
            "simulation_fps": 100,
            "control_freq": 20,
            "curriculum_config_path": "./curriculum_progress.json",
        },
        "algorithm": {
            "learning_rate": 3e-4,
            "n_steps": 2048,  # Longer rollouts for curriculum learning
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "net_arch": [256, 256],  # Larger networks for complex curriculum
            },
        },
        "training": {
            "total_timesteps": 2000000,  # 2M steps for full curriculum
            "save_freq": 50000,
            "log_interval": 1,
            "experiment_name": "curriculum_humanoid",
            "log_dir": "./logs/curriculum",
            "model_dir": "./models/curriculum",
        },
    }


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        print("Using default curriculum configuration")
        return create_curriculum_config()


def create_curriculum_env(env_config: Dict[str, Any], render_mode: str = None):
    """Create curriculum learning environment."""

    def _make_env():
        return CurriculumHumanoidEnv(
            curriculum_config_path=env_config.get("curriculum_config_path"),
            simulation_fps=env_config["simulation_fps"],
            control_freq=env_config["control_freq"],
            render_mode=render_mode,
        )

    # For curriculum learning, use single environment to maintain curriculum state
    env = Monitor(_make_env())
    print("✅ Created curriculum environment")

    return env


def train_curriculum(config_path: str = None, render: bool = False):
    """Train humanoid with curriculum learning."""

    # Load configuration
    config = load_config(config_path)
    env_config = config["env"]
    algo_config = config["algorithm"]
    training_config = config["training"]

    print("=== Curriculum Learning Training ===")
    print(f"Experiment: {training_config['experiment_name']}")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Curriculum config: {env_config.get('curriculum_config_path', 'default')}")

    # Create directories
    os.makedirs(training_config["log_dir"], exist_ok=True)
    os.makedirs(training_config["model_dir"], exist_ok=True)

    # Create environment
    render_mode = "human" if render else None
    env = create_curriculum_env(env_config, render_mode=render_mode)

    print(f"Environment observation space: {env.observation_space}")
    print(f"Environment action space: {env.action_space}")

    # Configure logger
    logger = configure(training_config["log_dir"], ["stdout", "tensorboard"])

    # Create PPO model
    print("\nCreating PPO model...")
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
        device="auto",
    )

    # Set custom logger
    model.set_logger(logger)

    # Create callbacks
    callbacks = []

    # Curriculum monitoring callback
    curriculum_callback = CurriculumCallback(log_freq=50, verbose=1)
    callbacks.append(curriculum_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config["save_freq"],
        save_path=os.path.join(training_config["model_dir"], "checkpoints"),
        name_prefix="curriculum_ppo",
    )
    callbacks.append(checkpoint_callback)

    # Combine callbacks
    callback = CallbackList(callbacks)

    try:
        print(f"\n🚀 Starting curriculum training...")
        print(f"Logging to: {training_config['log_dir']}")

        start_time = time.time()

        # Train the model
        model.learn(
            total_timesteps=training_config["total_timesteps"],
            callback=callback,
            log_interval=training_config["log_interval"],
            progress_bar=True,
        )

        training_time = time.time() - start_time

        # Save final model
        final_model_path = os.path.join(
            training_config["model_dir"], "final_curriculum_model"
        )
        model.save(final_model_path)
        print(f"\n✅ Training completed! Final model saved to: {final_model_path}")
        print(f"Training time: {training_time/3600:.2f} hours")

        # Save curriculum progress
        if hasattr(env, "save_curriculum_progress"):
            curriculum_path = os.path.join(
                training_config["model_dir"], "final_curriculum_progress.json"
            )
            env.save_curriculum_progress(curriculum_path)
            print(f"📊 Curriculum progress saved to: {curriculum_path}")

        # Print final curriculum status
        if hasattr(env, "get_curriculum_info"):
            final_info = env.get_curriculum_info()
            print(f"\n🎓 Final Curriculum Status:")
            print(f"   Final Stage: {final_info.get('current_stage', 'unknown')}")
            print(f"   Total Episodes: {final_info.get('total_episodes', 0)}")
            print(
                f"   Stage History: {len(final_info.get('stage_history', []))} completed stages"
            )

        # Quick evaluation
        print(f"\n🧪 Testing final model...")
        obs, _ = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(200):  # Test for 200 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break

        print(
            f"   Test episode: {steps} steps, {total_reward:.2f} reward, {total_reward/steps:.3f} avg"
        )

    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        interrupt_model_path = os.path.join(
            training_config["model_dir"], "interrupted_curriculum_model"
        )
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")

    finally:
        env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train humanoid with curriculum learning"
    )
    parser.add_argument(
        "--config", type=str, help="Path to training configuration file"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render during training (slower)"
    )

    args = parser.parse_args()

    train_curriculum(args.config, args.render)


if __name__ == "__main__":
    main()
