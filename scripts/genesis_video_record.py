#!/usr/bin/env python3
"""
Proper Genesis video recording using official Genesis API.
Based on Genesis repository documentation and examples.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stable_baselines3 import PPO
import genesis as gs


def main():
    parser = argparse.ArgumentParser(description="Record video using Genesis cameras")
    parser.add_argument(
        "--model", default="./models/test_sb3/final_model", help="Path to trained model"
    )
    parser.add_argument(
        "--output", default="genesis_robot_video.mp4", help="Output video filename"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of simulation steps"
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
        "--orbit", action="store_true", help="Orbit camera around robot"
    )
    args = parser.parse_args()

    print("=== Genesis Official Video Recording ===")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"Steps: {args.steps}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")

    # Load trained model
    print("\nLoading trained model...")
    model = PPO.load(args.model)
    print("✅ Model loaded!")

    # Initialize Genesis
    print("\nInitializing Genesis...")
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Create scene in headless mode for video recording
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.02,  # 50 FPS simulation
            substeps=10,
        ),
        show_viewer=False,  # Critical: no GUI for video recording
        renderer=gs.renderers.Rasterizer(),  # Better performance than ray tracer
    )

    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # Add robot
    project_root = Path(__file__).parent.parent
    urdf_path = project_root / "assets/robots/g1/g1_29dof.urdf"

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(urdf_path),
            pos=(0, 0, 1.0),
        )
    )

    # Add cameras for different angles
    print("Setting up cameras...")

    # Main camera (front view)
    cam_main = scene.add_camera(
        res=(args.resolution[0], args.resolution[1]),
        pos=(3.5, -2.0, 2.0),
        lookat=(0.0, 0.0, 0.8),
        fov=45,
        GUI=False,  # Critical for video recording
    )

    # Side camera
    cam_side = scene.add_camera(
        res=(args.resolution[0], args.resolution[1]),
        pos=(0.0, -4.0, 1.5),
        lookat=(0.0, 0.0, 0.8),
        fov=45,
        GUI=False,
    )

    print("Building scene...")
    scene.build()

    # Apply robot grounding
    print("Applying robot grounding...")
    sys.path.insert(0, str(project_root))
    from robot_grounding import RobotGroundingCalculator

    calculator = RobotGroundingCalculator(robot, verbose=False)
    grounding_height = calculator.get_grounding_height(safety_margin=0.03)
    robot.set_pos([0, 0, grounding_height])

    # Stabilize robot
    for _ in range(10):
        scene.step()

    print(f"✅ Robot loaded at height {grounding_height:.3f}m")

    # Start recording
    print(f"\nStarting video recording...")
    cam_main.start_recording()

    # Create environment wrapper for RL model
    print("Creating environment wrapper...")

    # Simple observation extraction
    def get_observation():
        pos = robot.get_pos().cpu().numpy()
        quat = robot.get_quat().cpu().numpy()
        joint_pos = robot.get_dofs_position().cpu().numpy()
        joint_vel = robot.get_dofs_velocity().cpu().numpy()

        # Dummy previous action and target velocity
        prev_action = np.zeros(len(joint_pos))
        target_vel = np.array([1.0])

        return np.concatenate(
            [pos, quat, joint_pos, joint_vel, prev_action, target_vel]
        )

    # Simulation and recording loop
    print(f"Recording {args.steps} steps...")
    total_reward = 0

    for step in range(args.steps):
        # Get observation
        obs = get_observation()

        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Apply action to robot (simple position control)
        current_pos = robot.get_dofs_position().cpu()  # Move to CPU first
        target_pos = current_pos + (action * 0.1)  # Small position changes
        robot.control_dofs_position(target_pos)

        # Step simulation
        scene.step()

        # Optional: orbit camera
        if args.orbit:
            angle = step * 2 * np.pi / args.steps
            radius = 4.0
            cam_x = radius * np.cos(angle)
            cam_y = radius * np.sin(angle)
            cam_main.set_pose(pos=(cam_x, cam_y, 2.5), lookat=(0.0, 0.0, 0.8))

        # Render frame (this captures it for the video)
        cam_main.render()

        # Progress update
        if step % 50 == 0:
            pos = robot.get_pos().cpu().numpy()
            print(f"  Step {step}: robot at [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Stop recording and save
    print(f"\nSaving video to {args.output}...")
    try:
        cam_main.stop_recording(save_to_filename=args.output, fps=args.fps)
        print("✅ Video saved successfully!")

        # Check file
        if os.path.exists(args.output):
            file_size = os.path.getsize(args.output) / (1024 * 1024)
            print(f"   File: {args.output}")
            print(f"   Size: {file_size:.2f} MB")
            print(f"   Duration: {args.steps/args.fps:.1f} seconds")
        else:
            print("⚠️  Video file not found - Genesis may have saved elsewhere")

    except Exception as e:
        print(f"❌ Error saving video: {e}")
        print("This can happen due to:")
        print("  - Missing video encoding libraries")
        print("  - GPU driver issues")
        print("  - File permission problems")

    print(f"\n🎬 To play the video:")
    print(f"   vlc {args.output}")
    print(f"   mpv {args.output}")
    print(f"   python -m http.server 8000  # Then open browser to localhost:8000")


if __name__ == "__main__":
    main()
