#!/usr/bin/env python3
"""
Record a video of the trained humanoid model performing walking tasks.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import os
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
from genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env
import genesis as gs


def main():
    parser = argparse.ArgumentParser(
        description="Record video of trained humanoid model"
    )
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--output", default="trained_humanoid_demo.mp4", help="Output video filename"
    )
    parser.add_argument(
        "--episodes", type=int, default=2, help="Number of episodes to record"
    )
    parser.add_argument(
        "--max-steps", type=int, default=150, help="Maximum steps per episode"
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    parser.add_argument(
        "--resolution",
        nargs=2,
        type=int,
        default=[1280, 720],
        help="Video resolution (width height)",
    )
    parser.add_argument(
        "--camera-distance", type=float, default=3.5, help="Camera distance from robot"
    )
    parser.add_argument(
        "--camera-height", type=float, default=2.0, help="Camera height"
    )
    args = parser.parse_args()

    print("=== Recording Trained Humanoid Model ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Episodes: {args.episodes}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"Video FPS: {args.fps}")

    # Load trained model
    print("\nLoading trained model...")
    model = PPO.load(args.model)
    print("✅ Model loaded successfully!")

    # Don't initialize Genesis here - let the environment handle it

    # Create output directory
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Recording setup
    frames_collected = 0
    all_frames = []
    episode_info = []

    # Run episodes
    for episode in range(args.episodes):
        print(f"\n--- Recording Episode {episode + 1}/{args.episodes} ---")

        # Create environment (new scene for each episode to ensure clean state)
        print("Creating environment...")
        env = make_humanoid_env(
            episode_length=args.max_steps,
            simulation_fps=50,
            control_freq=20,
            target_velocity=1.0,
            render_mode=None,  # We'll handle rendering manually
        )

        # Get the underlying environment and scene
        genesis_env = env.env  # Access wrapped environment
        scene = genesis_env.scene

        # Add camera for recording
        camera = scene.add_camera(
            res=(args.resolution[0], args.resolution[1]),
            pos=(args.camera_distance, -args.camera_distance, args.camera_height),
            lookat=(0.0, 0.0, 0.8),  # Look at robot center
            fov=45,
            GUI=False,
        )

        # Start recording for this episode
        camera.start_recording()

        # Reset environment
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        episode_frames = 0

        print(f"Recording episode {episode + 1}...")

        for step in range(args.max_steps):
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Render and capture frame
            camera.render(rgb=True)
            episode_frames += 1

            # Print progress
            if step % 50 == 0:
                print(f"  Step {step}: reward={reward:.3f}, total={total_reward:.2f}")

            # Check if episode ended
            if terminated or truncated:
                reason = "terminated" if terminated else "truncated"
                print(f"  Episode ended ({reason}) at step {step}")
                break

        # Stop recording for this episode
        temp_video = f"temp_episode_{episode}.mp4"
        camera.stop_recording(save_to_filename=temp_video, fps=args.fps)

        episode_info.append(
            {
                "episode": episode + 1,
                "steps": step_count,
                "total_reward": total_reward,
                "avg_reward": total_reward / step_count,
                "frames": episode_frames,
            }
        )

        print(f"Episode {episode + 1} complete:")
        print(f"  Total steps: {step_count}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Average reward: {total_reward/step_count:.3f}")
        print(f"  Frames recorded: {episode_frames}")

        # Clean up environment
        env.close()

        # Small delay between episodes
        if episode < args.episodes - 1:
            time.sleep(1)

    # Merge episode videos if multiple episodes
    if args.episodes > 1:
        print(f"\nMerging {args.episodes} episode videos...")
        # Use ffmpeg to concatenate videos
        concat_file = "concat_list.txt"
        with open(concat_file, "w") as f:
            for i in range(args.episodes):
                f.write(f"file 'temp_episode_{i}.mp4'\n")

        merge_cmd = (
            f"ffmpeg -f concat -safe 0 -i {concat_file} -c copy {args.output} -y"
        )
        os.system(merge_cmd)

        # Clean up temporary files
        os.remove(concat_file)
        for i in range(args.episodes):
            temp_file = f"temp_episode_{i}.mp4"
            if os.path.exists(temp_file):
                os.remove(temp_file)
    else:
        # Single episode, just rename
        os.rename("temp_episode_0.mp4", args.output)

    # Print summary
    print("\n=== Recording Summary ===")
    for info in episode_info:
        print(
            f"Episode {info['episode']}: {info['steps']} steps, "
            f"{info['total_reward']:.2f} reward, "
            f"{info['avg_reward']:.3f} avg/step"
        )

    total_frames = sum(info["frames"] for info in episode_info)
    print(f"\nTotal frames: {total_frames}")
    print(f"Video duration: {total_frames/args.fps:.1f} seconds")

    # Verify output
    if os.path.exists(args.output):
        file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
        print(f"\n✅ Video saved successfully!")
        print(f"   File: {args.output}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"\n🎬 To play the video:")
        print(f"   vlc {args.output}")
        print(f"   mpv {args.output}")
        print(f"   # or open in your video player")
    else:
        print(f"\n❌ Failed to create video file!")


if __name__ == "__main__":
    main()
