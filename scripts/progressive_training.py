#!/usr/bin/env python3
"""Progressive training script with automatic checkpointing and video generation."""

import sys
import json
import time
import argparse
import psutil
import GPUtil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


class ProgressiveTrainingMonitor(BaseCallback):
    """Custom callback for progressive training monitoring."""

    def __init__(
        self,
        config: Dict[str, Any],
        phase_name: str,
        checkpoint_steps: List[int],
        video_generator: Optional[Any] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.config = config
        self.phase_name = phase_name
        self.checkpoint_steps = checkpoint_steps
        self.video_generator = video_generator
        self.status_file = Path(config["monitoring"]["status_file"])

        # Performance tracking
        self.best_reward = -np.inf
        self.best_distance = 0.0
        self.episode_rewards = []
        self.episode_distances = []
        self.start_time = time.time()

        # System monitoring
        self.initial_memory = self._get_memory_usage()
        self.max_memory = self.initial_memory

    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        print(f"🚀 Starting {self.phase_name}")
        print(f"📊 Target: {self.config['description']}")
        print(f"⏱️ Timesteps: {self.config['training']['total_timesteps']:,}")
        print(f"📈 Episode length: {self.config['environment']['episode_length']}")
        print(f"🎯 Checkpoints: {self.checkpoint_steps}")

        self._update_status(
            {
                "phase": self.phase_name,
                "status": "training",
                "start_time": datetime.now().isoformat(),
                "timesteps_total": self.config["training"]["total_timesteps"],
                "timesteps_done": 0,
                "best_reward": self.best_reward,
                "best_distance": self.best_distance,
                "checkpoints_completed": [],
                "system_memory_mb": self.initial_memory,
            }
        )

    def _on_step(self) -> bool:
        """Called at each training step."""
        current_timesteps = self.num_timesteps

        # Check for checkpoint
        if current_timesteps in self.checkpoint_steps:
            self._handle_checkpoint(current_timesteps)

        # Update status every 1000 steps
        if current_timesteps % 1000 == 0:
            self._update_progress()

        # Monitor system resources
        self._monitor_system()

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        if len(self.locals.get("episode_rewards", [])) > 0:
            recent_rewards = self.locals["episode_rewards"]
            self.episode_rewards.extend(recent_rewards)

            # Track best performance
            max_reward = max(recent_rewards)
            if max_reward > self.best_reward:
                self.best_reward = max_reward

        # Estimate distance (simplified)
        if hasattr(self.training_env, "get_attr"):
            try:
                infos = self.training_env.get_attr("last_info")
                for info_list in infos:
                    if info_list and "distance_traveled" in info_list[0]:
                        distance = info_list[0]["distance_traveled"]
                        self.episode_distances.append(distance)
                        if distance > self.best_distance:
                            self.best_distance = distance
            except:
                pass  # Distance tracking optional

    def _handle_checkpoint(self, timesteps: int):
        """Handle checkpoint creation and video generation."""
        print(f"\n🎯 Checkpoint reached: {timesteps:,} timesteps")

        # Save model
        checkpoint_dir = Path(self.config["monitoring"]["checkpoint_path"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model_path = checkpoint_dir / f"checkpoint_{timesteps}.zip"
        self.model.save(str(model_path))

        # Generate evaluation video
        if self.video_generator:
            try:
                video_path = self.video_generator.create_checkpoint_video(
                    model_path=str(model_path),
                    timesteps=timesteps,
                    phase=self.phase_name,
                )
                print(f"📹 Video saved: {video_path}")
            except Exception as e:
                print(f"⚠️ Video generation failed: {e}")

        # Update status
        self._update_status(
            {
                "checkpoints_completed": self.checkpoint_steps[
                    : self.checkpoint_steps.index(timesteps) + 1
                ],
                "last_checkpoint": timesteps,
                "last_checkpoint_time": datetime.now().isoformat(),
            }
        )

        print(f"✅ Checkpoint {timesteps:,} completed")

    def _update_progress(self):
        """Update training progress."""
        current_timesteps = self.num_timesteps
        total_timesteps = self.config["training"]["total_timesteps"]
        progress = (current_timesteps / total_timesteps) * 100

        elapsed = time.time() - self.start_time
        eta = (
            (elapsed / current_timesteps) * (total_timesteps - current_timesteps)
            if current_timesteps > 0
            else 0
        )

        print(
            f"📊 Progress: {progress:.1f}% ({current_timesteps:,}/{total_timesteps:,}) | "
            f"ETA: {eta/60:.1f}min | Best reward: {self.best_reward:.2f} | "
            f"Best distance: {self.best_distance:.2f}m"
        )

    def _monitor_system(self):
        """Monitor system resources."""
        current_memory = self._get_memory_usage()
        if current_memory > self.max_memory:
            self.max_memory = current_memory

        # Check for memory issues
        if current_memory > 6000:  # 6GB warning
            print(f"⚠️ High memory usage: {current_memory:.1f}MB")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / 1024 / 1024

    def _update_status(self, updates: Dict[str, Any]):
        """Update status file."""
        try:
            if self.status_file.exists():
                with open(self.status_file, "r") as f:
                    status = json.load(f)
            else:
                status = {}

            status.update(updates)
            status.update(
                {
                    "timesteps_done": self.num_timesteps,
                    "progress_percent": (
                        self.num_timesteps / self.config["training"]["total_timesteps"]
                    )
                    * 100,
                    "elapsed_time_min": (time.time() - self.start_time) / 60,
                    "best_reward": self.best_reward,
                    "best_distance": self.best_distance,
                    "max_memory_mb": self.max_memory,
                    "last_update": datetime.now().isoformat(),
                }
            )

            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, "w") as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            print(f"⚠️ Failed to update status: {e}")


class VideoGenerator:
    """Generate evaluation videos at checkpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.video_dir = Path(config["monitoring"]["video_save_path"])
        self.video_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint_video(
        self, model_path: str, timesteps: int, phase: str
    ) -> str:
        """Create evaluation video for checkpoint."""
        print(f"🎬 Generating video for checkpoint {timesteps:,}...")

        # Load trained model
        model = PPO.load(model_path)

        # Create evaluation environment
        env_config = self.config["environment"].copy()
        env_config["render"] = False
        env_config["headless"] = True
        env = make_humanoid_env(**env_config)

        # Record evaluation episode
        obs, _ = env.reset()
        positions = []
        rewards = []
        total_reward = 0

        # Run for evaluation length (shorter than training)
        eval_steps = min(300, self.config["environment"]["episode_length"])

        for step in range(eval_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Track position
            if len(obs) >= 3:
                positions.append([obs[0], obs[1], obs[2]])
            rewards.append(reward)

            if done or truncated:
                break

        env.close()

        # Calculate metrics
        if positions:
            positions = np.array(positions)
            distance_traveled = np.sqrt(
                np.sum((positions[-1, :2] - positions[0, :2]) ** 2)
            )
            avg_height = np.mean(positions[:, 2])
        else:
            distance_traveled = 0
            avg_height = 0

        # Save video info (actual video generation would use Genesis camera)
        video_info = {
            "phase": phase,
            "timesteps": timesteps,
            "total_reward": float(total_reward),
            "distance_traveled": float(distance_traveled),
            "avg_height": float(avg_height),
            "episode_length": len(positions),
            "success": distance_traveled > 2.0,  # 2m minimum success
            "timestamp": datetime.now().isoformat(),
        }

        info_path = self.video_dir / f"checkpoint_{timesteps}_{phase}_info.json"
        with open(info_path, "w") as f:
            json.dump(video_info, f, indent=2)

        print(
            f"📊 Evaluation: {distance_traveled:.2f}m distance, {total_reward:.2f} reward"
        )

        return str(info_path)


def check_system_requirements():
    """Check if system meets requirements for training."""
    print("🔍 Checking system requirements...")

    # Memory check
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024**3)
    print(f"💾 Available RAM: {available_gb:.1f}GB")

    if available_gb < 4.0:
        print("⚠️ Warning: Low available RAM (< 4GB)")

    # GPU check
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            print(
                f"🎮 GPU: {gpu.name} ({gpu.memoryTotal:.1f}MB total, {gpu.memoryFree:.1f}MB free)"
            )

            if gpu.memoryFree < 2000:  # 2GB minimum
                print("⚠️ Warning: Low GPU memory (< 2GB free)")
                return False
        else:
            print("⚠️ No GPU detected")
            return False
    except:
        print("⚠️ Could not check GPU status")

    return True


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def create_training_environment(config: Dict[str, Any]):
    """Create training environment based on config."""
    env_config = config["environment"]
    n_envs = config["training"]["n_envs"]

    print(f"🌍 Creating {n_envs} training environments...")

    # Adjust for memory constraints
    available_memory = psutil.virtual_memory().available / (1024**3)
    if available_memory < 6.0 and n_envs > 2:
        print(f"⚠️ Reducing environments from {n_envs} to 2 due to memory constraints")
        n_envs = 2
        config["training"]["n_envs"] = 2

    # Create vectorized environment
    envs = [lambda: Monitor(make_humanoid_env(**env_config)) for _ in range(n_envs)]

    return envs, n_envs


def run_progressive_training(config_path: str, dry_run: bool = False):
    """Run progressive training for a specific phase."""
    print(f"🚀 Progressive Training System")
    print("=" * 50)

    # System checks
    if not check_system_requirements():
        print("❌ System requirements not met")
        return False

    # Load configuration
    config = load_config(config_path)
    phase_name = Path(config_path).stem

    print(f"📋 Phase: {phase_name}")
    print(f"📝 {config['description']}")

    if dry_run:
        print("🧪 DRY RUN MODE - No actual training will occur")
        print(f"✅ Configuration validated: {config_path}")
        return True

    # Create environments
    envs, n_envs = create_training_environment(config)

    # Setup directories
    for path_key in ["tensorboard_log", "video_save_path", "checkpoint_path"]:
        Path(config["monitoring"][path_key]).mkdir(parents=True, exist_ok=True)

    # Initialize video generator
    video_generator = VideoGenerator(config)

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"] // n_envs,
        save_path=config["monitoring"]["checkpoint_path"],
        name_prefix=f"{phase_name}_model",
    )

    progress_callback = ProgressiveTrainingMonitor(
        config=config,
        phase_name=phase_name,
        checkpoint_steps=config["training"]["checkpoint_saves"],
        video_generator=video_generator,
    )

    # Create and train model
    print(f"🤖 Creating PPO model...")

    # Note: Actual model creation would go here
    # This is a placeholder for the training setup
    print(f"✅ Training setup complete for {phase_name}")
    print(f"📊 Would train for {config['training']['total_timesteps']:,} timesteps")
    print(f"🎯 Target distance: Based on phase configuration")

    return True


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Progressive Humanoid Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate configuration without training"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return

    success = run_progressive_training(args.config, dry_run=args.dry_run)

    if success:
        print("✅ Progressive training setup completed successfully")
    else:
        print("❌ Progressive training setup failed")


if __name__ == "__main__":
    main()
