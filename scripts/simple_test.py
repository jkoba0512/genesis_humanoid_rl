#!/usr/bin/env python3
"""
Simple test script to verify the environment works without SB3 overhead.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env
import numpy as np


def main():
    print("=== Simple Environment Test ===")

    # Create environment with minimal settings
    env = make_humanoid_env(
        episode_length=10,  # Very short episode
        simulation_fps=50,
        control_freq=10,
        target_velocity=1.0,
    )

    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    print("\nTesting environment reset...")
    obs, info = env.reset()
    print(f"Reset successful. Observation shape: {obs.shape}")

    # Test a few steps
    print("\nTesting environment steps...")
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            print("Episode ended early")
            break

    print(f"\nTest completed successfully!")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final observation shape: {obs.shape}")


if __name__ == "__main__":
    main()
