#!/usr/bin/env python3
"""Real progressive training with actual PPO learning."""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


class ProgressCallback(BaseCallback):
    """Progress monitoring callback."""

    def __init__(self, phase_name: str, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.phase_name = phase_name
        self.save_path = Path(save_path)
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        # Log progress every 1000 steps
        if self.num_timesteps % 1000 == 0:
            print(f"📊 {self.phase_name}: {self.num_timesteps:,} steps completed")

        return True

    def _on_rollout_end(self) -> None:
        if len(self.locals.get("episode_rewards", [])) > 0:
            recent_rewards = self.locals["episode_rewards"]
            recent_lengths = self.locals["episode_lengths"]

            self.episode_rewards.extend(recent_rewards)
            self.episode_lengths.extend(recent_lengths)

            # Track best performance
            max_reward = max(recent_rewards)
            if max_reward > self.best_reward:
                self.best_reward = max_reward
                print(f"🏆 New best reward: {max_reward:.2f}")


def create_env(config: Dict[str, Any], rank: int = 0):
    """Create single environment."""

    def _init():
        env = make_humanoid_env(**config["environment"])
        env = Monitor(env)
        return env

    set_random_seed(rank)
    return _init


def run_real_training(config_path: str):
    """Run real PPO training."""
    print(f"🚀 Real Training Started")
    print("=" * 50)

    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    phase_name = Path(config_path).stem
    print(f"📋 Phase: {phase_name}")
    print(f"📝 {config['description']}")
    print(f"⏱️ Timesteps: {config['training']['total_timesteps']:,}")
    print(f"📈 Episode length: {config['environment']['episode_length']}")

    # Create directories
    log_dir = Path(config["monitoring"]["tensorboard_log"])
    checkpoint_dir = Path(config["monitoring"]["checkpoint_path"])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create environments
    n_envs = config["training"]["n_envs"]
    print(f"🌍 Creating {n_envs} environments...")

    # Create vectorized environments
    if n_envs == 1:
        env = DummyVecEnv([create_env(config, 0)])
    else:
        env = SubprocVecEnv([create_env(config, i) for i in range(n_envs)])

    print(f"✅ Environments created")

    # Create PPO model
    print(f"🤖 Creating PPO model...")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["ppo"]["learning_rate"],
        n_steps=config["training"]["n_steps"],
        batch_size=config["training"]["batch_size"],
        n_epochs=config["ppo"]["n_epochs"],
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        clip_range=config["ppo"]["clip_range"],
        ent_coef=config["ppo"]["ent_coef"],
        vf_coef=config["ppo"]["vf_coef"],
        max_grad_norm=config["ppo"]["max_grad_norm"],
        policy_kwargs=config["network"]["policy_kwargs"],
        tensorboard_log=str(log_dir),
        verbose=1,
        device="auto",
    )

    print(f"✅ PPO model created")

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config["training"]["save_freq"] // n_envs,
        save_path=str(checkpoint_dir),
        name_prefix=f"{phase_name}_checkpoint",
    )

    progress_callback = ProgressCallback(
        phase_name=phase_name, save_path=str(checkpoint_dir)
    )

    # Start training
    print(
        f"🔥 Starting training for {config['training']['total_timesteps']:,} timesteps..."
    )
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=config["training"]["total_timesteps"],
            callback=[checkpoint_callback, progress_callback],
            tb_log_name=phase_name,
            reset_num_timesteps=True,
            progress_bar=True,
        )

        elapsed = time.time() - start_time
        print(f"✅ Training completed in {elapsed/60:.1f} minutes")

        # Save final model
        final_model_path = checkpoint_dir / f"{phase_name}_final_model.zip"
        model.save(str(final_model_path))
        print(f"💾 Final model saved: {final_model_path}")

        # Evaluate final performance
        print(f"🎯 Evaluating final performance...")
        obs = env.reset()
        total_reward = 0
        episode_rewards = []

        # Run 5 evaluation episodes
        for episode in range(5):
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < config["environment"]["episode_length"]:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += (
                    reward[0] if isinstance(reward, np.ndarray) else reward
                )
                step_count += 1

                if done:
                    obs = env.reset()
                    break

            episode_rewards.append(episode_reward)
            print(
                f"  Episode {episode + 1}: {episode_reward:.2f} reward, {step_count} steps"
            )

        avg_reward = np.mean(episode_rewards)
        print(f"📊 Average evaluation reward: {avg_reward:.2f}")

        # Save results
        results = {
            "phase": phase_name,
            "training_timesteps": config["training"]["total_timesteps"],
            "training_time_minutes": elapsed / 60,
            "final_model_path": str(final_model_path),
            "evaluation": {
                "episodes": len(episode_rewards),
                "rewards": episode_rewards,
                "average_reward": float(avg_reward),
                "best_reward": float(max(episode_rewards)),
            },
            "completion_time": datetime.now().isoformat(),
        }

        results_path = checkpoint_dir / f"{phase_name}_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"📋 Results saved: {results_path}")

        env.close()
        return True, results

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        env.close()
        return False, {"error": "interrupted"}

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        env.close()
        return False, {"error": str(e)}


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Real progressive training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )

    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"❌ Configuration file not found: {args.config}")
        return

    success, results = run_real_training(args.config)

    if success:
        print("✅ Training completed successfully")
        print(f"🎯 Average reward: {results['evaluation']['average_reward']:.2f}")
    else:
        print("❌ Training failed")


if __name__ == "__main__":
    main()
