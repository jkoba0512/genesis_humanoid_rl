#!/usr/bin/env python3
"""
Demo script to run the trained humanoid model with visualization.
"""

import sys
from pathlib import Path
import argparse

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env
import time


def main():
    parser = argparse.ArgumentParser(description="Demo trained humanoid model")
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--episodes", type=int, default=3, help="Number of episodes to run"
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="Simulation speed multiplier"
    )
    args = parser.parse_args()

    print("=== Humanoid Walking Demo ===")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")

    # Load trained model
    print("\nLoading trained model...")
    model = PPO.load(args.model)
    print("✅ Model loaded successfully!")

    # Create environment with rendering
    print("\nCreating environment...")
    env = make_humanoid_env(
        episode_length=args.max_steps,
        simulation_fps=50,
        control_freq=20,
        target_velocity=1.0,
        render_mode="human",  # Enable visualization
    )
    print("✅ Environment created!")

    # Run episodes
    for episode in range(args.episodes):
        print(f"\n--- Episode {episode + 1}/{args.episodes} ---")

        obs, _ = env.reset()
        total_reward = 0
        step_count = 0

        for step in range(args.max_steps):
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Add delay for visualization
            if args.speed > 0:
                time.sleep(0.02 / args.speed)  # ~50 FPS base rate

            # Print progress every 50 steps
            if step % 50 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.2f}")

            # Check if episode ended
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"  Episode ended ({reason}) at step {step}")
                break

        print(f"Episode {episode + 1} complete:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward: {total_reward/step_count:.3f}")

        if episode < args.episodes - 1:
            print("Press Enter to continue to next episode...")
            input()

    env.close()
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    main()
