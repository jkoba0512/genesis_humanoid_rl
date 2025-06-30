#!/usr/bin/env python3
"""Simple video recording of robot walking."""

import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def create_walking_video():
    """Create video of robot walking with trajectory visualization."""
    print("🎬 Creating Robot Walking Video")
    print("=" * 50)

    # Create environment
    env_config = {
        "episode_length": 400,
        "simulation_fps": 30,
        "control_freq": 10,
        "render": False,
        "headless": True,
        "target_velocity": 1.0,
    }

    env = make_humanoid_env(**env_config)
    print("✅ Environment created")

    # Record robot movement
    print("🔴 Recording robot movement...")
    obs, _ = env.reset()

    positions = []
    heights = []
    rewards = []
    actions_history = []
    timestamps = []

    total_reward = 0
    start_time = time.time()

    for step in range(400):
        # Simple walking pattern
        t = step * 0.03
        action = np.zeros(env.action_space.shape[0])

        # Enhanced walking pattern
        if len(action) >= 20:
            # Hip movements with more realistic pattern
            action[6] = 0.2 * np.sin(t) + 0.05 * np.sin(3 * t)  # Left hip
            action[7] = -0.2 * np.sin(t) - 0.05 * np.sin(3 * t)  # Right hip

            # Knee movements
            action[12] = 0.15 * np.sin(t + np.pi / 2) + 0.1  # Left knee
            action[13] = 0.15 * np.sin(t - np.pi / 2) + 0.1  # Right knee

            # Ankle movements for balance
            action[18] = 0.08 * np.sin(t + np.pi / 4)  # Left ankle
            action[19] = -0.08 * np.sin(t - np.pi / 4)  # Right ankle

            # Arm movements for balance
            if len(action) > 25:
                action[0] = 0.1 * np.sin(t + np.pi)  # Left arm counter to right leg
                action[1] = -0.1 * np.sin(t + np.pi)  # Right arm counter to left leg

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Record data
        if len(obs) >= 3:
            positions.append([obs[0], obs[1]])
            heights.append(obs[2])
        else:
            positions.append([0, 0])
            heights.append(0)

        rewards.append(reward)
        actions_history.append(np.mean(np.abs(action)))
        timestamps.append(step * 0.033)  # 30 FPS

        if done or truncated:
            print(f"Episode ended at step {step}")
            break

        if step % 50 == 0:
            print(
                f"  Step {step}/400 | Reward: {total_reward:.2f} | Position: [{obs[0]:.2f}, {obs[1]:.2f}]"
            )

    elapsed = time.time() - start_time
    env.close()

    # Convert to arrays
    positions = np.array(positions)
    heights = np.array(heights)
    rewards = np.array(rewards)
    actions_history = np.array(actions_history)
    timestamps = np.array(timestamps)

    print(f"✅ Recording completed:")
    print(f"   Duration: {elapsed:.1f}s")
    print(f"   Total reward: {total_reward:.2f}")
    print(f"   Distance: {np.sqrt(np.sum((positions[-1] - positions[0])**2)):.2f}m")
    print(f"   Average height: {np.mean(heights):.2f}m")

    # Create animated visualization
    print("🎨 Creating animated video...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("🤖 Humanoid Robot Learning to Walk", fontsize=16, fontweight="bold")

    # Setup subplots
    # 1. Robot trajectory (top-left)
    ax1.set_xlim(positions[:, 0].min() - 0.5, positions[:, 0].max() + 0.5)
    ax1.set_ylim(positions[:, 1].min() - 0.5, positions[:, 1].max() + 0.5)
    ax1.set_xlabel("X Position (m)")
    ax1.set_ylabel("Y Position (m)")
    ax1.set_title("🚶 Robot Walking Path")
    ax1.grid(True, alpha=0.3)

    # 2. Height over time (top-right)
    ax2.set_xlim(0, len(heights))
    ax2.set_ylim(heights.min() - 0.1, heights.max() + 0.1)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Height (m)")
    ax2.set_title("📏 Robot Height (Balance)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.8, color="g", linestyle="--", alpha=0.5, label="Target Height")
    ax2.legend()

    # 3. Cumulative reward (bottom-left)
    cumulative_rewards = np.cumsum(rewards)
    ax3.set_xlim(0, len(rewards))
    ax3.set_ylim(cumulative_rewards.min(), cumulative_rewards.max())
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Cumulative Reward")
    ax3.set_title("📈 Learning Progress")
    ax3.grid(True, alpha=0.3)

    # 4. Action magnitude (bottom-right)
    ax4.set_xlim(0, len(actions_history))
    ax4.set_ylim(0, actions_history.max() * 1.1)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Action Magnitude")
    ax4.set_title("🎛️ Control Effort")
    ax4.grid(True, alpha=0.3)

    # Animation elements
    (line1,) = ax1.plot([], [], "b-", linewidth=2, label="Path")
    robot_pos = Circle((0, 0), 0.1, color="red", alpha=0.8)
    ax1.add_patch(robot_pos)
    ax1.legend()

    (line2,) = ax2.plot([], [], "g-", linewidth=2)
    (line3,) = ax3.plot([], [], "purple", linewidth=2)
    (line4,) = ax4.plot([], [], "orange", linewidth=2)

    # Animation function
    def animate(frame):
        # Update trajectory
        if frame > 0:
            line1.set_data(positions[:frame, 0], positions[:frame, 1])
            robot_pos.center = (positions[frame - 1, 0], positions[frame - 1, 1])

            # Update height
            line2.set_data(range(frame), heights[:frame])

            # Update reward
            line3.set_data(range(frame), cumulative_rewards[:frame])

            # Update actions
            line4.set_data(range(frame), actions_history[:frame])

        return line1, robot_pos, line2, line3, line4

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(positions), interval=100, blit=True, repeat=True
    )

    # Save video
    output_path = Path("./videos/robot_walking_analysis.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving video to {output_path}...")

    # Use FFMpegWriter for better quality
    Writer = animation.writers["ffmpeg"]
    writer = Writer(fps=15, metadata=dict(artist="Genesis Humanoid RL"), bitrate=1800)

    anim.save(str(output_path), writer=writer)

    plt.close()

    print(f"✅ Video created successfully!")
    print(f"   📁 File: {output_path}")
    print(f"   📏 Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   🎬 Duration: {len(positions) / 15:.1f} seconds")
    print(f"   🖼️ Resolution: 1200x1000")

    # Create summary image
    summary_path = output_path.with_suffix(".png")

    plt.figure(figsize=(15, 10))

    # Plot final results
    plt.subplot(2, 3, 1)
    plt.plot(positions[:, 0], positions[:, 1], "b-", linewidth=2)
    plt.scatter(
        positions[0, 0], positions[0, 1], color="green", s=100, label="Start", zorder=5
    )
    plt.scatter(
        positions[-1, 0], positions[-1, 1], color="red", s=100, label="End", zorder=5
    )
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("🚶 Complete Walking Path")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis("equal")

    plt.subplot(2, 3, 2)
    plt.plot(heights, "g-", linewidth=2)
    plt.axhline(y=0.8, color="r", linestyle="--", alpha=0.7, label="Target")
    plt.xlabel("Time Step")
    plt.ylabel("Height (m)")
    plt.title("📏 Height Stability")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(cumulative_rewards, "purple", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Cumulative Reward")
    plt.title("📈 Learning Progress")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(rewards, "orange", linewidth=1, alpha=0.7)
    plt.xlabel("Time Step")
    plt.ylabel("Instant Reward")
    plt.title("⚡ Reward per Step")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(actions_history, "brown", linewidth=2)
    plt.xlabel("Time Step")
    plt.ylabel("Action Magnitude")
    plt.title("🎛️ Control Effort")
    plt.grid(True, alpha=0.3)

    # Summary statistics
    plt.subplot(2, 3, 6)
    plt.axis("off")
    stats_text = f"""
🤖 Robot Performance Summary

📏 Distance Traveled: {np.sqrt(np.sum((positions[-1] - positions[0])**2)):.2f}m
📊 Total Reward: {total_reward:.2f}
📈 Average Reward: {np.mean(rewards):.3f}
🏃 Average Height: {np.mean(heights):.2f}m
⏱️ Episode Length: {len(positions)} steps
🎯 Success: {'✅ Forward Motion' if positions[-1, 0] > positions[0, 0] else '❌ No Progress'}
    """
    plt.text(
        0.1,
        0.5,
        stats_text,
        fontsize=12,
        verticalalignment="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(str(summary_path), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"📊 Summary image: {summary_path}")

    return str(output_path), str(summary_path)


if __name__ == "__main__":
    video_path, summary_path = create_walking_video()
    print(f"\n🎬 Video creation completed!")
    print(f"🎥 Video: {video_path}")
    print(f"📊 Summary: {summary_path}")
