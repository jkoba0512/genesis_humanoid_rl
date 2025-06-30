#!/usr/bin/env python3
"""Train humanoid robot with checkpoints for learning evolution video."""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


class EvolutionCallback(BaseCallback):
    """Callback for tracking learning evolution and creating checkpoints."""

    def __init__(
        self,
        save_path,
        checkpoint_steps,
        status_file="evolution_status.json",
        verbose=1,
    ):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_steps = checkpoint_steps
        self.status_file = status_file
        self.start_time = time.time()
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_training_start(self):
        """Initialize status tracking."""
        self._update_status("training", 0)
        print("\n🤖 Starting Humanoid Learning Evolution Training")
        print(f"📊 TensorBoard: tensorboard --logdir {self.logger.dir} --host 0.0.0.0")
        print(f"💾 Checkpoints will be saved at: {self.checkpoint_steps}")
        print("=" * 60)

    def _on_step(self) -> bool:
        """Track progress and save at checkpoints."""
        # Collect episode data
        for i, done in enumerate(self.locals["dones"]):
            if done and "episode" in self.locals["infos"][i]:
                self.episode_rewards.append(self.locals["infos"][i]["episode"]["r"])
                self.episode_lengths.append(self.locals["infos"][i]["episode"]["l"])

        # Update status
        current_step = self.num_timesteps
        self._update_status("training", current_step)

        # Save checkpoint at specific steps
        if current_step in self.checkpoint_steps:
            checkpoint_path = self.save_path / f"checkpoint_{current_step}"
            self.model.save(checkpoint_path)
            print(f"\n💾 Checkpoint saved at step {current_step}: {checkpoint_path}")

            # Log statistics
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"📈 Mean Reward (last 100 eps): {mean_reward:.2f}")
                print(f"📏 Mean Episode Length: {mean_length:.2f}")

        # Progress update every 1000 steps
        if current_step % 1000 == 0:
            elapsed = time.time() - self.start_time
            steps_per_sec = current_step / elapsed
            remaining_steps = self.locals["total_timesteps"] - current_step
            eta = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

            print(
                f"\r⏱️  Step {current_step}/{self.locals['total_timesteps']} "
                f"({current_step/self.locals['total_timesteps']*100:.1f}%) "
                f"| Speed: {steps_per_sec:.0f} steps/s "
                f"| ETA: {eta/60:.1f} min",
                end="",
            )

        return True

    def _update_status(self, status, current_steps):
        """Update training status file."""
        status_data = {
            "status": status,
            "current_steps": current_steps,
            "total_steps": getattr(self, "locals", {}).get("total_timesteps", 100000),
            "progress": (current_steps / 100000) * 100,
            "checkpoint_steps": self.checkpoint_steps,
            "last_update": datetime.now().isoformat(),
        }

        with open(self.status_file, "w") as f:
            json.dump(status_data, f, indent=2)

    def _on_training_end(self):
        """Mark training as complete."""
        self._update_status("completed", self.num_timesteps)
        print(
            f"\n\n✅ Training Complete! Total time: {(time.time()-self.start_time)/60:.1f} minutes"
        )
        print(f"📁 All checkpoints saved in: {self.save_path}")


def train_evolution():
    """Main training function for learning evolution."""
    # Load configuration
    config_path = Path("configs/learning_evolution.json")
    with open(config_path) as f:
        config = json.load(f)

    # Set up directories
    experiment_name = config["training"]["experiment_name"]
    log_dir = Path(f"./logs/{experiment_name}")
    model_dir = Path(f"./models/{experiment_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("🏗️  Setting up training environment...")

    # Create vectorized environment
    n_envs = config["env"]["n_envs"]
    env_config = config["env"].copy()
    env_config.pop("n_envs")

    def make_env(rank):
        def _init():
            env = make_humanoid_env(**env_config)
            env = Monitor(env)
            return env

        return _init

    if n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0)])

    print(f"✅ Created {n_envs} parallel environments")

    # Configure logger
    logger = configure(str(log_dir), ["tensorboard", "stdout"])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["ppo"]["n_steps"],
        batch_size=config["ppo"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        policy_kwargs=config["ppo"]["policy_kwargs"],
        verbose=1,
        seed=config["training"]["seed"],
        device=config["training"]["device"],
        tensorboard_log=str(log_dir),
    )
    model.set_logger(logger)

    print("✅ PPO model created")

    # Create callbacks
    evolution_callback = EvolutionCallback(
        save_path=model_dir, checkpoint_steps=config["training"]["checkpoint_saves"]
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"],
        save_path=model_dir,
        name_prefix="rl_model",
    )

    callbacks = CallbackList([evolution_callback, checkpoint_callback])

    # Start training
    print("\n🚀 Starting training with learning evolution tracking...")
    print(f"📊 Total timesteps: {config['training']['total_timesteps']:,}")
    print(f"💾 Checkpoints at: {config['training']['checkpoint_saves']}")
    print("=" * 60 + "\n")

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=callbacks,
            log_interval=config["training"]["log_interval"],
            progress_bar=False,  # We use custom progress
        )

        # Save final model
        final_path = model_dir / "final_model"
        model.save(final_path)
        print(f"\n💾 Final model saved: {final_path}")

        # Create completion marker
        with open("EVOLUTION_COMPLETE.txt", "w") as f:
            f.write(f"Training completed at {datetime.now()}\n")
            f.write(f"Model saved at: {final_path}\n")
            f.write(f"Checkpoints: {config['training']['checkpoint_saves']}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        model.save(model_dir / f"interrupted_model_{model.num_timesteps}")

    finally:
        env.close()


if __name__ == "__main__":
    train_evolution()
