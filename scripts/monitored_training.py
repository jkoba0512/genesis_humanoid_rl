#!/usr/bin/env python3
"""
Run training with monitoring and completion notification.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def run_monitored_training():
    """Run training with progress monitoring."""

    # Configuration for medium-length training
    config = {
        "total_timesteps": 50000,  # ~30-45 minutes
        "n_envs": 2,
        "save_freq": 10000,
        "log_dir": "./logs/monitored_training",
        "model_dir": "./models/monitored_training",
    }

    print("=== 🤖 Starting Monitored Robot Training ===")
    print(f"Total steps: {config['total_timesteps']:,}")
    print(f"Estimated time: 30-45 minutes")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 50)

    # Create directories
    os.makedirs(config["log_dir"], exist_ok=True)
    os.makedirs(config["model_dir"], exist_ok=True)

    # Create completion marker file
    status_file = "training_status.json"

    # Write initial status
    with open(status_file, "w") as f:
        json.dump(
            {
                "status": "running",
                "start_time": datetime.now().isoformat(),
                "total_steps": config["total_timesteps"],
                "current_steps": 0,
                "progress": 0,
            },
            f,
        )

    try:
        # Create environment
        print("\n📋 Creating environment...")
        env = make_humanoid_env(
            episode_length=500,
            simulation_fps=100,
            control_freq=20,
            target_velocity=1.0,
            n_envs=config["n_envs"],
        )

        # Configure logger
        logger = configure(config["log_dir"], ["stdout", "tensorboard"])

        # Create model
        print("🧠 Creating PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            policy_kwargs={"net_arch": [256, 256, 128]},
            verbose=1,
            device="auto",
        )

        model.set_logger(logger)

        print("\n🚀 Training started!")
        print("📊 Monitor progress at: http://100.101.234.88:6007")
        print("🌐 Dashboard at: http://100.101.234.88:8080/progress_dashboard.html")
        print("\n" + "=" * 50 + "\n")

        start_time = time.time()

        # Custom callback to update status
        class StatusCallback:
            def __init__(self, status_file, total_steps):
                self.status_file = status_file
                self.total_steps = total_steps
                self.last_update = 0

            def __call__(self, locals_dict, globals_dict):
                current_steps = locals_dict.get("self").num_timesteps

                # Update every 1000 steps
                if current_steps - self.last_update >= 1000:
                    self.last_update = current_steps
                    progress = (current_steps / self.total_steps) * 100

                    with open(self.status_file, "w") as f:
                        json.dump(
                            {
                                "status": "running",
                                "start_time": datetime.now().isoformat(),
                                "total_steps": self.total_steps,
                                "current_steps": current_steps,
                                "progress": progress,
                                "estimated_remaining": self._estimate_remaining(
                                    current_steps
                                ),
                            },
                            f,
                        )

                    if current_steps % 5000 == 0:
                        print(
                            f"\n📊 Progress: {progress:.1f}% ({current_steps:,}/{self.total_steps:,} steps)"
                        )

                return True

            def _estimate_remaining(self, current_steps):
                if current_steps == 0:
                    return "Calculating..."
                elapsed = time.time() - start_time
                rate = current_steps / elapsed
                remaining_steps = self.total_steps - current_steps
                remaining_time = remaining_steps / rate
                return str(timedelta(seconds=int(remaining_time)))

        # Train with callback
        callback = StatusCallback(status_file, config["total_timesteps"])

        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback,
            log_interval=10,
            progress_bar=True,
        )

        # Save final model
        final_model_path = os.path.join(config["model_dir"], "final_model")
        model.save(final_model_path)

        # Training complete!
        elapsed_time = time.time() - start_time

        # Write completion status
        with open(status_file, "w") as f:
            json.dump(
                {
                    "status": "completed",
                    "start_time": datetime.now().isoformat(),
                    "total_steps": config["total_timesteps"],
                    "current_steps": config["total_timesteps"],
                    "progress": 100,
                    "elapsed_time": str(timedelta(seconds=int(elapsed_time))),
                    "model_path": final_model_path,
                },
                f,
            )

        print("\n" + "=" * 50)
        print("🎉 TRAINING COMPLETE! 🎉")
        print("=" * 50)
        print(f"\n✅ Total time: {timedelta(seconds=int(elapsed_time))}")
        print(f"✅ Final model saved: {final_model_path}")
        print(f"✅ TensorBoard logs: {config['log_dir']}")
        print("\n📊 View results at: http://100.101.234.88:6007")
        print("🎬 You can now create videos of your trained robot!")
        print("\nTo evaluate the trained model:")
        print(
            f"  uv run python scripts/evaluate_sb3.py {final_model_path} --render --episodes 5"
        )

        # Create completion notification file
        with open("TRAINING_COMPLETE.txt", "w") as f:
            f.write(
                f"Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total training time: {timedelta(seconds=int(elapsed_time))}\n")
            f.write(f"Model saved at: {final_model_path}\n")

    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
        with open(status_file, "w") as f:
            json.dump(
                {"status": "interrupted", "message": "Training stopped by user"}, f
            )
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        with open(status_file, "w") as f:
            json.dump({"status": "error", "message": str(e)}, f)
    finally:
        env.close()


if __name__ == "__main__":
    run_monitored_training()
