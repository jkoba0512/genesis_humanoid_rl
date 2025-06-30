#!/usr/bin/env python3
"""
Training script that generates progression videos at checkpoints.
Shows how the robot learns to walk over time.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.logger import configure

from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


class ProgressionVideoCallback(BaseCallback):
    """Callback that creates videos at specific checkpoints."""

    def __init__(
        self, save_freq: int, video_dir: str, video_length: int = 200, verbose: int = 1
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.video_dir = video_dir
        self.video_length = video_length
        self.checkpoint_num = 0
        os.makedirs(video_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if we should create a video
        if self.n_calls % self.save_freq == 0:
            self._create_progression_video()
        return True

    def _create_progression_video(self):
        """Create a video showing current performance."""
        print(f"\n📹 Creating progression video at step {self.n_calls}...")

        # Create filename based on training progress
        stage_name = self._get_stage_name()
        video_path = os.path.join(
            self.video_dir, f"stage_{self.checkpoint_num}_{stage_name}.mp4"
        )

        # Import video recording function
        try:
            from genesis_humanoid_rl.utils.video_recorder import record_policy_video

            # Record video of current policy
            record_policy_video(
                self.model,
                self.training_env,
                video_path,
                n_steps=self.video_length,
                deterministic=True,
            )

            print(f"✅ Video saved: {video_path}")

        except ImportError:
            print("⚠️ Video recording not available, creating placeholder")
            # Create a placeholder to continue training
            with open(video_path + ".txt", "w") as f:
                f.write(f"Video placeholder for checkpoint {self.checkpoint_num}")

        self.checkpoint_num += 1

    def _get_stage_name(self):
        """Get descriptive name for current training stage."""
        steps = self.n_calls
        if steps == 0:
            return "untrained"
        elif steps <= 25000:
            return "early_learning"
        elif steps <= 50000:
            return "discovering_walking"
        elif steps <= 75000:
            return "refining_gait"
        else:
            return "smooth_walking"


def train_with_progression_videos(config_path: str):
    """Train humanoid with automatic progression video generation."""

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    env_config = config["env"]
    algo_config = config["algorithm"]
    training_config = config["training"]

    print("=== Progressive Training with Video Generation ===")
    print(f"Total timesteps: {training_config['total_timesteps']:,}")
    print(f"Video checkpoints every: {training_config['save_freq']:,} steps")
    print(f"This will generate videos showing learning progression\n")

    # Create directories
    os.makedirs(training_config["log_dir"], exist_ok=True)
    os.makedirs(training_config["model_dir"], exist_ok=True)
    video_dir = os.path.join(training_config["model_dir"], "progression_videos")
    os.makedirs(video_dir, exist_ok=True)

    # Create environment
    print("Creating training environment...")
    env = make_humanoid_env(
        episode_length=env_config["episode_length"],
        simulation_fps=env_config["simulation_fps"],
        control_freq=env_config["control_freq"],
        target_velocity=env_config["target_velocity"],
        n_envs=env_config["n_envs"],
    )

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
        device="auto",
    )

    model.set_logger(logger)

    # First, create video of untrained robot
    print("\n📹 Creating initial video of untrained robot...")
    initial_video_path = os.path.join(video_dir, "stage_0_untrained.mp4")

    # Simple evaluation of untrained model
    obs, _ = env.reset()
    frames = []
    rewards = []

    for _ in range(200):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    print(f"Untrained robot average reward: {np.mean(rewards):.3f}")
    print(f"Note: Videos will be generated using Genesis recording at checkpoints")

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=training_config["save_freq"],
        save_path=os.path.join(training_config["model_dir"], "checkpoints"),
        name_prefix="progression_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Video progression callback
    video_callback = ProgressionVideoCallback(
        save_freq=training_config["save_freq"],
        video_dir=video_dir,
        video_length=200,
        verbose=1,
    )
    callbacks.append(video_callback)

    # Combine callbacks
    callback = CallbackList(callbacks)

    # Training
    try:
        print(f"\n🚀 Starting progressive training...")
        print(f"Videos will be saved to: {video_dir}")
        print(f"Models will be saved to: {training_config['model_dir']}")
        print(f"\nExpected videos:")
        print(f"  - Stage 0: Untrained (random movements)")
        print(f"  - Stage 1: After 25k steps (early learning)")
        print(f"  - Stage 2: After 50k steps (discovering walking)")
        print(f"  - Stage 3: After 75k steps (refining gait)")
        print(f"  - Stage 4: After 100k steps (smooth walking)")

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
            training_config["model_dir"], "final_progression_model"
        )
        model.save(final_model_path)

        print(f"\n✅ Training completed!")
        print(f"Training time: {training_time/3600:.2f} hours")
        print(f"Final model saved to: {final_model_path}")
        print(f"\n📹 Videos saved in: {video_dir}")
        print(f"You can now create a montage showing the learning progression!")

    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        interrupt_model_path = os.path.join(
            training_config["model_dir"], "interrupted_model"
        )
        model.save(interrupt_model_path)
        print(f"Model saved to: {interrupt_model_path}")

    finally:
        env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train humanoid with progression videos"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/progression_demo.json",
        help="Path to training configuration file",
    )

    args = parser.parse_args()

    # Check if config exists
    if not os.path.exists(args.config):
        print(f"❌ Config file not found: {args.config}")
        print("Using default progression_demo.json config")
        args.config = "configs/progression_demo.json"

    train_with_progression_videos(args.config)


if __name__ == "__main__":
    main()
