#!/usr/bin/env python3
"""
Simple frame capture script for creating videos of the trained model.
Uses matplotlib to create frames if Genesis recording fails.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def main():
    parser = argparse.ArgumentParser(
        description="Simple video capture of trained model"
    )
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--output", default="humanoid_animation.mp4", help="Output video filename"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of steps to record"
    )
    parser.add_argument("--fps", type=int, default=20, help="Video frames per second")
    args = parser.parse_args()

    print("=== Simple Video Capture ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Steps: {args.steps}")

    # Load model
    model = PPO.load(args.model)
    print("✅ Model loaded!")

    # Create environment
    env = make_humanoid_env(
        episode_length=args.steps,
        simulation_fps=50,
        control_freq=20,
        target_velocity=1.0,
    )
    print("✅ Environment created!")

    # Collect trajectory data
    print("\nCollecting trajectory data...")
    obs, _ = env.reset()

    # Storage for visualization data
    positions = []
    rewards = []
    actions = []

    for step in range(args.steps):
        # Get action
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)

        # Extract position data from observation
        # obs format: [pos(3), quat(4), joint_pos(35), joint_vel(35), prev_action(35), target_vel(1)]
        robot_pos = obs[:3]  # x, y, z position

        positions.append(robot_pos)
        rewards.append(reward)
        actions.append(action)

        if step % 50 == 0:
            print(
                f"  Step {step}: pos=[{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}], reward={reward:.3f}"
            )

        if terminated or truncated:
            print(f"  Episode ended at step {step}")
            break

    env.close()

    # Convert to numpy arrays
    positions = np.array(positions)
    rewards = np.array(rewards)
    actions = np.array(actions)

    print(f"\nCollected {len(positions)} frames of data")
    print(f"Total reward: {np.sum(rewards):.2f}")

    # Create animated plot
    print("\nCreating animated visualization...")

    fig = plt.figure(figsize=(12, 8))

    # Subplot 1: 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.set_xlabel("X Position")
    ax1.set_ylabel("Y Position")
    ax1.set_zlabel("Z Position")
    ax1.set_title("Robot Trajectory")

    # Subplot 2: Height over time
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Height (m)")
    ax2.set_title("Robot Height")
    ax2.set_ylim(0, 1.5)

    # Subplot 3: Reward over time
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Reward")
    ax3.set_title("Step Rewards")

    # Subplot 4: Action magnitudes
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Action Magnitude")
    ax4.set_title("Control Signal Strength")

    # Animation function
    def animate(frame):
        # Clear axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        # Plot 3D trajectory up to current frame
        ax1.plot(
            positions[:frame, 0],
            positions[:frame, 1],
            positions[:frame, 2],
            "b-",
            alpha=0.7,
        )
        ax1.scatter(
            positions[frame - 1, 0],
            positions[frame - 1, 1],
            positions[frame - 1, 2],
            c="red",
            s=100,
            marker="o",
        )
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.set_zlabel("Z Position")
        ax1.set_title(f"Robot Trajectory (Step {frame})")

        # Plot height
        ax2.plot(range(frame), positions[:frame, 2], "g-")
        ax2.axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="Target height")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Height (m)")
        ax2.set_title("Robot Height")
        ax2.set_xlim(0, len(positions))
        ax2.set_ylim(0, 1.5)
        ax2.grid(True, alpha=0.3)

        # Plot rewards
        ax3.plot(range(frame), rewards[:frame], "b-")
        ax3.fill_between(range(frame), rewards[:frame], alpha=0.3)
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Reward")
        ax3.set_title(f"Cumulative Reward: {np.sum(rewards[:frame]):.2f}")
        ax3.set_xlim(0, len(rewards))
        ax3.grid(True, alpha=0.3)

        # Plot action magnitudes
        action_mags = np.linalg.norm(actions[:frame], axis=1)
        ax4.plot(range(frame), action_mags, "r-")
        ax4.set_xlabel("Step")
        ax4.set_ylabel("||action||")
        ax4.set_title("Control Signal Strength")
        ax4.set_xlim(0, len(actions))
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(positions), interval=1000 / args.fps, blit=False
    )

    # Save animation
    print(f"Saving video to {args.output}...")
    writer = FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(args.output, writer=writer)

    print(f"\n✅ Video saved successfully!")
    print(f"   File: {args.output}")
    print(f"   Duration: {len(positions)/args.fps:.1f} seconds")
    print(f"\n🎬 To play the video:")
    print(f"   vlc {args.output}")
    print(f"   mpv {args.output}")

    # Also save a summary plot
    summary_file = args.output.replace(".mp4", "_summary.png")
    animate(len(positions) - 1)  # Show final frame
    plt.savefig(summary_file, dpi=150, bbox_inches="tight")
    print(f"\n📊 Summary plot saved to: {summary_file}")


if __name__ == "__main__":
    main()
