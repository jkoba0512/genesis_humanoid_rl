#!/usr/bin/env python3
"""
Record video of trained model using Genesis camera recording.
Fixed version that handles Genesis initialization properly.
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
import genesis as gs


def create_environment_manually(
    episode_length=200, simulation_fps=50, control_freq=20, target_velocity=1.0
):
    """Create environment manually to have full control over Genesis scene and camera."""

    # Import after path setup
    from genesis_humanoid_rl.environments.humanoid_env import HumanoidWalkingEnv
    from genesis_humanoid_rl.environments.sb3_wrapper import SB3HumanoidEnv

    # Create the base environment
    base_env = HumanoidWalkingEnv(
        render_mode=None,  # We'll add our own camera
        simulation_fps=simulation_fps,
        control_freq=control_freq,
        episode_length=episode_length,
        target_velocity=target_velocity,
    )

    # Wrap for SB3 compatibility
    env = SB3HumanoidEnv(base_env)

    return env, base_env


def main():
    parser = argparse.ArgumentParser(
        description="Record video of trained humanoid model"
    )
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--output", default="trained_robot_demo.mp4", help="Output video filename"
    )
    parser.add_argument(
        "--steps", type=int, default=150, help="Number of steps to record"
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
    parser.add_argument(
        "--camera-orbit", action="store_true", help="Make camera orbit around robot"
    )
    args = parser.parse_args()

    print("=== Genesis Video Recording ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Steps: {args.steps}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"Video FPS: {args.fps}")

    # Load trained model
    print("\nLoading trained model...")
    model = PPO.load(args.model)
    print("✅ Model loaded successfully!")

    # Create output directory
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    print("\nCreating environment...")
    # Create environment with manual control
    env, base_env = create_environment_manually(
        episode_length=args.steps,
        simulation_fps=50,
        control_freq=20,
        target_velocity=1.0,
    )

    # Reset environment to initialize scene
    obs, _ = env.reset()

    # Get the Genesis scene from the base environment
    scene = base_env.scene

    print("Adding camera to scene...")
    # Add camera for recording
    camera = scene.add_camera(
        res=(args.resolution[0], args.resolution[1]),
        pos=(args.camera_distance, -args.camera_distance, args.camera_height),
        lookat=(0.0, 0.0, 0.8),  # Look at robot center
        fov=45,
        GUI=False,  # Headless mode
    )

    print("Starting video recording...")
    # Start recording
    camera.start_recording()

    # Recording loop
    total_reward = 0
    step_count = 0

    print(f"\nRecording {args.steps} steps...")
    start_time = time.time()

    for step in range(args.steps):
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=True)

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1

        # Optionally orbit camera around robot
        if args.camera_orbit:
            angle = step * 2 * np.pi / args.steps  # Full rotation over episode
            camera_x = args.camera_distance * np.cos(angle)
            camera_y = args.camera_distance * np.sin(angle)
            camera.set_pose(
                pos=(camera_x, camera_y, args.camera_height), lookat=(0.0, 0.0, 0.8)
            )

        # Render and capture frame
        camera.render(rgb=True)

        # Progress update
        if step % 50 == 0 and step > 0:
            elapsed = time.time() - start_time
            fps = step / elapsed
            print(
                f"  Step {step}/{args.steps}: reward={reward:.3f}, total={total_reward:.2f}, {fps:.1f} steps/s"
            )

        # Check if episode ended
        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"\nEpisode ended ({reason}) at step {step}")
            break

    # Stop recording and save video
    print(f"\nSaving video to {args.output}...")
    try:
        camera.stop_recording(save_to_filename=args.output, fps=args.fps)
        print("✅ Video saved successfully!")
    except Exception as e:
        print(f"❌ Error saving video: {e}")
        # Try alternative filename
        alt_output = args.output.replace(".mp4", "_backup.mp4")
        print(f"Trying alternative filename: {alt_output}")
        try:
            camera.stop_recording(save_to_filename=alt_output, fps=args.fps)
            print(f"✅ Video saved to: {alt_output}")
            args.output = alt_output
        except:
            print("❌ Failed to save video")

    # Clean up
    env.close()

    # Summary
    print("\n=== Recording Summary ===")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward: {total_reward/step_count:.3f}")
    print(f"Recording time: {time.time() - start_time:.1f}s")

    # Verify output
    if os.path.exists(args.output):
        file_size = os.path.getsize(args.output) / (1024 * 1024)  # MB
        print(f"\n✅ Video file created!")
        print(f"   File: {args.output}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"   Duration: ~{step_count/args.fps:.1f} seconds")
        print(f"\n🎬 To play the video:")
        print(f"   vlc {args.output}")
        print(f"   mpv {args.output}")
        print(f"   ffplay {args.output}")
    else:
        print(f"\n⚠️  Video file not found at expected location")
        print("Genesis may have saved it elsewhere or recording failed")


if __name__ == "__main__":
    main()
