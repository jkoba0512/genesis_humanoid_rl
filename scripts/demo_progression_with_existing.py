#!/usr/bin/env python3
"""
Demonstrate robot learning progression using existing resources.
This creates a simulated progression to show what the learning process looks like.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def create_learning_curve_animation():
    """Create an animated learning curve showing progression."""

    # Simulated learning data
    steps = np.linspace(0, 100000, 500)

    # Reward curves for different stages
    untrained = np.random.normal(-50, 10, len(steps)) - 50
    early_learning = -50 * np.exp(-steps / 20000) + np.random.normal(0, 5, len(steps))
    discovering = (
        20 * (1 - np.exp(-steps / 30000)) + np.random.normal(0, 8, len(steps)) - 20
    )
    smooth_walking = (
        80 * (1 - np.exp(-steps / 40000)) + np.random.normal(0, 5, len(steps)) - 10
    )

    # Combine stages
    reward = np.zeros_like(steps)
    for i, step in enumerate(steps):
        if step < 10000:
            reward[i] = untrained[i]
        elif step < 30000:
            weight = (step - 10000) / 20000
            reward[i] = (1 - weight) * untrained[i] + weight * early_learning[i]
        elif step < 60000:
            weight = (step - 30000) / 30000
            reward[i] = (1 - weight) * early_learning[i] + weight * discovering[i]
        else:
            weight = (step - 60000) / 40000
            reward[i] = (1 - weight) * discovering[i] + weight * smooth_walking[i]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Humanoid Robot Learning to Walk - Progression", fontsize=16)

    # Learning curve plot
    ax1.set_xlim(0, 100000)
    ax1.set_ylim(-100, 100)
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Episode Reward")
    ax1.grid(True, alpha=0.3)

    # Add stage labels
    ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax1.text(
        5000,
        -80,
        "Untrained\n(Falls)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
    )
    ax1.text(
        25000,
        -30,
        "Early Learning\n(Balance)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.3),
    )
    ax1.text(
        50000,
        20,
        "Discovering\n(First Steps)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
    )
    ax1.text(
        80000,
        60,
        "Smooth Walking\n(Mastery)",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
    )

    # Robot visualization
    ax2.set_xlim(-2, 8)
    ax2.set_ylim(0, 2)
    ax2.set_aspect("equal")
    ax2.set_xlabel("Distance (meters)")
    ax2.set_ylabel("Height (meters)")

    # Ground line
    ax2.plot([-2, 8], [0, 0], "k-", linewidth=2)

    # Initialize plot elements
    (line,) = ax1.plot([], [], "b-", linewidth=2, label="Reward")
    (point,) = ax1.plot([], [], "ro", markersize=8)

    # Robot representation (simplified)
    robot_body = patches.Rectangle((0, 0.5), 0.3, 0.6, facecolor="blue")
    robot_head = patches.Circle((0.15, 1.2), 0.15, facecolor="lightblue")
    robot_leg1 = patches.Rectangle((0.05, 0), 0.1, 0.5, facecolor="darkblue")
    robot_leg2 = patches.Rectangle((0.15, 0), 0.1, 0.5, facecolor="darkblue")

    ax2.add_patch(robot_body)
    ax2.add_patch(robot_head)
    ax2.add_patch(robot_leg1)
    ax2.add_patch(robot_leg2)

    # Progress text
    progress_text = ax2.text(4, 1.5, "", fontsize=12, ha="center")

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return (
            line,
            point,
            robot_body,
            robot_head,
            robot_leg1,
            robot_leg2,
            progress_text,
        )

    def animate(frame):
        # Update learning curve
        idx = frame * 5  # Speed up animation
        if idx < len(steps):
            line.set_data(steps[:idx], reward[:idx])
            point.set_data([steps[idx]], [reward[idx]])

            # Update robot position based on performance
            current_reward = reward[idx]
            current_step = steps[idx]

            # Robot behavior based on stage
            if current_step < 10000:  # Untrained - falling
                x_pos = 0
                body_angle = np.sin(frame * 0.5) * 30  # Wobbling
                body_y = 0.5 - abs(np.sin(frame * 0.3)) * 0.3
                progress_text.set_text("Stage 1: Learning Balance")
            elif current_step < 30000:  # Early learning - standing
                x_pos = 0.5
                body_angle = np.sin(frame * 0.2) * 10  # Small wobble
                body_y = 0.5
                progress_text.set_text("Stage 2: First Steps")
            elif current_step < 60000:  # Discovering - walking slowly
                x_pos = 2 + (current_step - 30000) / 30000 * 2
                body_angle = np.sin(frame * 0.1) * 5
                body_y = 0.5 + abs(np.sin(frame * 0.2)) * 0.05
                progress_text.set_text("Stage 3: Walking Emerges")
            else:  # Smooth walking
                x_pos = 4 + (current_step - 60000) / 40000 * 3
                body_angle = np.sin(frame * 0.1) * 2
                body_y = 0.5 + abs(np.sin(frame * 0.3)) * 0.02
                progress_text.set_text("Stage 4: Smooth Walking!")

            # Update robot position
            robot_body.set_x(x_pos)
            robot_body.set_y(body_y)
            robot_head.center = (x_pos + 0.15, body_y + 0.7)

            # Leg animation (walking motion)
            leg_phase = frame * 0.2
            robot_leg1.set_x(x_pos + 0.05 + np.sin(leg_phase) * 0.05)
            robot_leg2.set_x(x_pos + 0.15 + np.sin(leg_phase + np.pi) * 0.05)

        return (
            line,
            point,
            robot_body,
            robot_head,
            robot_leg1,
            robot_leg2,
            progress_text,
        )

    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init, frames=200, interval=50, blit=True, repeat=True
    )

    plt.tight_layout()

    # Save animation
    output_dir = "./models/progression_demo/progression_videos"
    os.makedirs(output_dir, exist_ok=True)

    print("💾 Saving learning progression animation...")
    anim.save(
        os.path.join(output_dir, "learning_progression_animation.gif"),
        writer="pillow",
        fps=20,
    )

    # Also save as static stages
    fig2, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig2.suptitle("Robot Learning Stages", fontsize=16)

    stages = [
        (0, "Stage 1: Untrained\n(Falls immediately)", -80),
        (25000, "Stage 2: Early Learning\n(Learning balance)", -20),
        (50000, "Stage 3: Discovering Walking\n(First steps)", 20),
        (100000, "Stage 4: Smooth Walking\n(Natural gait)", 70),
    ]

    for idx, (step_val, title, reward_val) in enumerate(stages):
        ax = axes[idx // 2, idx % 2]

        # Plot learning curve up to this point
        mask = steps <= step_val
        ax.plot(steps[mask], reward[mask], "b-", linewidth=2)
        ax.plot(step_val, reward_val, "ro", markersize=10)

        ax.set_xlim(0, 100000)
        ax.set_ylim(-100, 100)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Episode Reward")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_stages_comparison.png"), dpi=150)

    print("✅ Saved learning progression visualizations!")
    print(f"📁 Files saved in: {output_dir}")
    print("  - learning_progression_animation.gif")
    print("  - learning_stages_comparison.png")

    # Create summary statistics
    print("\n📊 Learning Progression Summary:")
    print("=" * 50)
    print("Stage 1 (0-10k steps): Average reward: -80 (Robot falls)")
    print("Stage 2 (10k-30k steps): Average reward: -20 (Learning balance)")
    print("Stage 3 (30k-60k steps): Average reward: +20 (First walking)")
    print("Stage 4 (60k-100k steps): Average reward: +70 (Smooth walking)")
    print("=" * 50)

    plt.show()


def explain_existing_video():
    """Explain what the existing video shows."""

    print("\n🎬 About the Existing Robot Video")
    print("=" * 50)
    print("The file 'genesis_robot_video.mp4' shows:")
    print("- A Unitree G1 humanoid robot with 35 degrees of freedom")
    print("- The robot attempting to walk in a physics simulation")
    print("- Genesis physics engine providing realistic dynamics")
    print("\nWhat you're seeing in the video:")
    print("1. Robot starts in standing position")
    print("2. Attempts to maintain balance")
    print("3. Takes steps forward (if trained)")
    print("4. May fall if not fully trained")
    print("\nTo see the full learning progression:")
    print("1. Complete the 100k step training (~60-75 minutes)")
    print("2. Videos will be automatically generated at checkpoints")
    print("3. You'll see clear progression from falling to walking")
    print("=" * 50)


def main():
    """Create progression demonstration."""

    print("🤖 Humanoid Robot Learning Progression Demo")
    print("=" * 50)

    # Check if training is running
    training_running = os.path.exists("training_progression.log")

    if training_running:
        print("📊 Training is currently in progress!")
        print("This demo will create visualizations of the expected progression.")
    else:
        print("💡 This demo shows what the learning progression looks like.")

    # Create visualizations
    create_learning_curve_animation()

    # Explain existing video
    explain_existing_video()

    print("\n✅ Demo complete!")
    print("\n📺 Monitor your actual training progress at:")
    print("   http://localhost:6007 (TensorBoard)")


if __name__ == "__main__":
    main()
