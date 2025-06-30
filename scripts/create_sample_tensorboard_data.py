#!/usr/bin/env python3
"""
Create sample TensorBoard data to demonstrate what successful training looks like.
"""

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


def create_sample_training_data():
    """Generate realistic training data for TensorBoard."""

    log_dir = "./logs/sample_training"
    os.makedirs(log_dir, exist_ok=True)

    print("📊 Creating sample TensorBoard data...")

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir)

    # Simulate 10,000 training steps
    total_steps = 10000

    # Generate realistic learning curves
    for step in range(0, total_steps, 100):
        # Episode reward (starts negative, improves over time)
        base_reward = -50 * np.exp(-step / 2000) + 50 * (1 - np.exp(-step / 5000))
        episode_reward = base_reward + np.random.normal(0, 10)

        # Episode length (gets longer as robot learns to balance)
        base_length = 50 + 950 * (1 - np.exp(-step / 3000))
        episode_length = base_length + np.random.normal(0, 50)
        episode_length = max(10, min(1000, episode_length))

        # Policy loss (decreases over time)
        policy_loss = 0.5 * np.exp(-step / 5000) + 0.01 + np.random.normal(0, 0.01)

        # Value loss (also decreases)
        value_loss = 1.0 * np.exp(-step / 4000) + 0.05 + np.random.normal(0, 0.02)

        # Learning rate (constant or decreasing)
        learning_rate = 3e-4 * (1 - step / total_steps)

        # Write to TensorBoard
        writer.add_scalar("rollout/ep_rew_mean", episode_reward, step)
        writer.add_scalar("rollout/ep_len_mean", episode_length, step)
        writer.add_scalar("train/policy_gradient_loss", policy_loss, step)
        writer.add_scalar("train/value_loss", value_loss, step)
        writer.add_scalar("train/learning_rate", learning_rate, step)
        writer.add_scalar(
            "train/entropy_loss", -0.01 * (1 + 0.1 * np.random.randn()), step
        )
        writer.add_scalar("train/approx_kl", 0.02 * (1 + 0.2 * np.random.randn()), step)
        writer.add_scalar(
            "train/clip_fraction", 0.1 * (1 + 0.3 * np.random.randn()), step
        )
        writer.add_scalar("time/fps", 200 + 50 * np.random.randn(), step)
        writer.add_scalar("time/iterations", step / 2048, step)
        writer.add_scalar("time/total_timesteps", step, step)

        # Add some custom metrics
        writer.add_scalar(
            "custom/forward_distance", step / 100 * (1 + 0.1 * np.random.randn()), step
        )
        writer.add_scalar(
            "custom/stability_score",
            min(1.0, step / 5000 + 0.1 * np.random.randn()),
            step,
        )

    # Close writer
    writer.close()

    print(f"✅ Sample data created in: {log_dir}")
    print("\n📊 This simulates a successful training session where:")
    print("  - Episode rewards improve from -50 to +50")
    print("  - Episode length increases (robot stays upright longer)")
    print("  - Policy and value losses decrease")
    print("  - Robot learns to walk forward")

    return log_dir


def main():
    """Create sample data and provide instructions."""

    log_dir = create_sample_training_data()

    print("\n🎯 Next Steps:")
    print("1. TensorBoard has been restarted with sample data")
    print("2. Visit: http://100.101.234.88:6007")
    print("3. You should now see:")
    print("   - SCALARS tab with multiple graphs")
    print("   - Learning curves showing improvement")
    print("   - Training metrics and performance data")
    print("\n💡 This is what your actual training will look like!")
    print("   Real training takes longer to generate this data.")


if __name__ == "__main__":
    main()
