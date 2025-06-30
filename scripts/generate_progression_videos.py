#!/usr/bin/env python3
"""
Generate progression videos from saved checkpoints.
Creates individual videos and a combined montage showing learning progression.
"""

import os
import sys
from pathlib import Path
import subprocess
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env
import genesis as gs


def generate_checkpoint_video(checkpoint_path, output_path, steps=500, stage_name=""):
    """Generate a video from a specific checkpoint."""

    print(f"\n📹 Generating video for: {stage_name}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_path}")

    # Initialize Genesis
    gs.init()

    # Create environment
    env = make_humanoid_env(
        episode_length=1000,
        simulation_fps=100,
        control_freq=20,
        target_velocity=1.0,
        render_mode=None,  # Headless for video recording
    )

    # Load model if checkpoint exists
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading trained model...")
        model = PPO.load(checkpoint_path, env=env)
    else:
        print(f"   Using untrained model...")
        model = PPO("MlpPolicy", env, verbose=0)

    # Create scene for video recording
    scene = gs.Scene(
        sim_options=gs.SimOptions(
            dt=1.0 / 100,
            substeps=2,
        ),
        renderer_options=gs.renderers.RenderOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=False,
    )

    # Add ground and robot
    plane = scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="/home/jkoba/SynologyDrive/genesis_humanoid_rl/assets/robots/g1/g1_29dof.urdf",
            pos=(0, 0, 0.787),
        ),
    )

    # Build scene
    scene.build()

    # Setup camera for recording
    camera = scene.add_camera(
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
        fov=40,
        resolution=(1280, 720),
        GUI=False,
    )

    # Run simulation and record
    print(f"   Recording {steps} steps...")

    obs, _ = env.reset()

    total_reward = 0
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0

    for step in range(steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        current_episode_reward += reward
        current_episode_length += 1

        # Apply action to Genesis robot
        if hasattr(robot, "set_dofs_position"):
            robot.set_dofs_position(position=action * 0.1)

        # Step physics
        scene.step()

        # Render frame
        camera.render()

        # Handle episode end
        if terminated or truncated:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
            current_episode_reward = 0
            current_episode_length = 0
            obs, _ = env.reset()

            # Reset robot position
            robot.set_pos((0, 0, 0.787))
            robot.set_quat((1, 0, 0, 0))

    # Save video
    print(f"   Saving video...")
    camera.stop_recording(save_path=output_path)

    # Calculate statistics
    avg_reward = (
        np.mean(episode_rewards)
        if episode_rewards
        else current_episode_reward / max(current_episode_length, 1)
    )
    avg_length = np.mean(episode_lengths) if episode_lengths else current_episode_length

    print(f"   ✅ Video saved!")
    print(f"   Average reward: {avg_reward:.2f}")
    print(f"   Average episode length: {avg_length:.1f}")

    # Cleanup
    env.close()

    return avg_reward, avg_length


def create_progression_montage(video_dir, output_path):
    """Create a side-by-side montage of all progression videos."""

    print(f"\n🎬 Creating progression montage...")

    # Find all stage videos
    stage_videos = sorted(
        [
            f
            for f in os.listdir(video_dir)
            if f.startswith("stage_") and f.endswith(".mp4")
        ]
    )

    if len(stage_videos) < 2:
        print("❌ Not enough videos for montage")
        return

    # Create montage using ffmpeg
    if len(stage_videos) == 4:
        # 2x2 grid for 4 videos
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            os.path.join(video_dir, stage_videos[0]),
            "-i",
            os.path.join(video_dir, stage_videos[1]),
            "-i",
            os.path.join(video_dir, stage_videos[2]),
            "-i",
            os.path.join(video_dir, stage_videos[3]),
            "-filter_complex",
            "[0:v]scale=640:360[v0];"
            "[1:v]scale=640:360[v1];"
            "[2:v]scale=640:360[v2];"
            "[3:v]scale=640:360[v3];"
            "[v0][v1]hstack[top];"
            "[v2][v3]hstack[bottom];"
            "[top][bottom]vstack[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            output_path,
        ]
    else:
        # Horizontal stack for any number of videos
        inputs = []
        scales = []
        for i, video in enumerate(stage_videos[:4]):  # Max 4 videos
            inputs.extend(["-i", os.path.join(video_dir, video)])
            scales.append(f"[{i}:v]scale=320:180[v{i}]")

        filter_str = ";".join(scales) + ";"
        filter_str += (
            "".join([f"[v{i}]" for i in range(len(stage_videos[:4]))])
            + f"hstack=inputs={len(stage_videos[:4])}[out]"
        )

        cmd = (
            ["ffmpeg", "-y"]
            + inputs
            + [
                "-filter_complex",
                filter_str,
                "-map",
                "[out]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                output_path,
            ]
        )

    print(f"Creating montage with {len(stage_videos)} videos...")
    subprocess.run(cmd)

    print(f"✅ Montage saved to: {output_path}")


def main():
    """Generate all progression videos."""

    model_dir = "./models/progression_demo"
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    video_dir = os.path.join(model_dir, "progression_videos")

    os.makedirs(video_dir, exist_ok=True)

    print("=== Generating Robot Learning Progression Videos ===")

    # Define checkpoints and their descriptions
    checkpoints = [
        (None, "stage_0_untrained.mp4", "Untrained Robot"),
        (
            os.path.join(checkpoint_dir, "progression_model_25000_steps.zip"),
            "stage_1_early_learning.mp4",
            "Early Learning (25k steps)",
        ),
        (
            os.path.join(checkpoint_dir, "progression_model_50000_steps.zip"),
            "stage_2_discovering_walking.mp4",
            "Discovering Walking (50k steps)",
        ),
        (
            os.path.join(checkpoint_dir, "progression_model_75000_steps.zip"),
            "stage_3_refining_gait.mp4",
            "Refining Gait (75k steps)",
        ),
        (
            os.path.join(checkpoint_dir, "progression_model_100000_steps.zip"),
            "stage_4_smooth_walking.mp4",
            "Smooth Walking (100k steps)",
        ),
    ]

    # Generate individual videos
    results = []
    for checkpoint_path, output_name, stage_name in checkpoints:
        output_path = os.path.join(video_dir, output_name)

        # Skip if already exists
        if os.path.exists(output_path):
            print(f"\n⏩ Skipping {stage_name} (already exists)")
            continue

        try:
            avg_reward, avg_length = generate_checkpoint_video(
                checkpoint_path, output_path, steps=500, stage_name=stage_name
            )
            results.append((stage_name, avg_reward, avg_length))
        except Exception as e:
            print(f"❌ Error generating video for {stage_name}: {e}")

    # Create montage
    montage_path = os.path.join(video_dir, "learning_progression_montage.mp4")
    create_progression_montage(video_dir, montage_path)

    # Print summary
    print("\n=== Summary ===")
    print("Generated videos:")
    for name, reward, length in results:
        print(f"  - {name}: Avg Reward={reward:.2f}, Avg Length={length:.1f}")

    print(f"\nAll videos saved in: {video_dir}")
    print(f"Montage: {montage_path}")

    # Create a simple HTML viewer
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Robot Learning Progression</title>
    <style>
        body { font-family: Arial; margin: 20px; background: #f0f0f0; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; }
        h1 { color: #333; }
        .video-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; }
        .video-item { text-align: center; }
        video { width: 100%; max-width: 500px; }
        .stats { margin-top: 10px; font-size: 14px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Humanoid Robot Learning Progression</h1>
        <p>Watch how the robot learns to walk through reinforcement learning!</p>
        
        <h2>Full Progression Montage</h2>
        <video controls>
            <source src="learning_progression_montage.mp4" type="video/mp4">
        </video>
        
        <h2>Individual Stages</h2>
        <div class="video-grid">
"""

    for checkpoint_path, output_name, stage_name in checkpoints:
        html_content += f"""
            <div class="video-item">
                <h3>{stage_name}</h3>
                <video controls>
                    <source src="{output_name}" type="video/mp4">
                </video>
            </div>
"""

    html_content += """
        </div>
    </div>
</body>
</html>
"""

    with open(os.path.join(video_dir, "view_progression.html"), "w") as f:
        f.write(html_content)

    print(f"\n🌐 Open view_progression.html in your browser to see all videos!")


if __name__ == "__main__":
    main()
