#!/usr/bin/env python3
"""
Basic example of using the humanoid walking environment.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from genesis_humanoid_rl.environments.humanoid_env import HumanoidWalkingEnv


def main():
    """Run basic example."""
    print("Creating humanoid walking environment...")
    
    # Create environment
    env = HumanoidWalkingEnv(
        render_mode="human",  # Enable visualization
        simulation_fps=100,
        control_freq=20,
        episode_length=1000,
        target_velocity=1.0,
    )
    
    print(f"Environment created!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run random actions
    print("\nRunning random actions for 100 steps...")
    total_reward = 0.0
    
    for step in range(100):
        # Sample random action
        action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render
        env.render()
        
        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            break
        
        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}: reward = {reward:.3f}, total = {total_reward:.3f}")
    
    print(f"\nFinal total reward: {total_reward:.3f}")
    
    # Cleanup
    env.close()
    print("Example completed!")


if __name__ == "__main__":
    main()