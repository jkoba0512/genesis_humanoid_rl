#!/usr/bin/env python3
"""Genesis camera-based video recording using official Genesis recording API."""

import sys
import numpy as np
from pathlib import Path
import time
import os
from datetime import datetime
import genesis as gs

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.physics.robot_grounding import RobotGroundingFactory


def record_genesis_video():
    """Record video using Genesis camera system with official recording API."""
    print("🎬 Genesis Camera Video Recording")
    print("=" * 50)

    # Initialize Genesis
    print("Initializing Genesis...")
    gs.init(backend=gs.cuda, logging_level="warning")

    # Create scene for video recording
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=1 / 60,  # 60 FPS simulation
            substeps=4,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            max_FPS=60,
            camera_pos=(4.0, -2.0, 2.0),
            camera_lookat=(0.0, 0.0, 1.0),
            camera_fov=45,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
        ),
        show_viewer=False,  # Headless for video recording
    )

    print("✅ Genesis scene created")

    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # Load G1 robot with automatic grounding
    robot_path = Path("assets/robots/g1/g1_29dof.urdf")
    if not robot_path.exists():
        print(f"❌ Robot file not found: {robot_path}")
        return None

    # Use fallback robot height for now
    robot_height = 0.787  # G1 robot standard grounding height
    print(f"🤖 Robot grounding height: {robot_height:.3f}m")

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_path),
            pos=(0, 0, robot_height),
            euler=(0, 0, 0),
        ),
    )

    print("🤖 G1 robot loaded")

    # Setup recording camera BEFORE building scene
    camera = scene.add_camera(
        res=(1280, 720),
        pos=(3.0, -1.5, 1.5),  # Side-front view
        lookat=(0.0, 0.0, robot_height),  # Focus on robot center
        fov=40,
        GUI=False,  # Essential for headless recording
    )

    print("📷 Recording camera configured")

    # Build scene (required in Genesis v0.2.1)
    print("Building scene...")
    scene.build()
    print(f"✅ Scene built - Robot DOFs: {robot.n_dofs}")

    # Prepare video output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_dir = Path("./videos")
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"genesis_robot_recording_{timestamp}.mp4"

    print(f"🎥 Video will be saved to: {video_path}")

    # Recording phases for comprehensive demonstration
    phases = [
        {"name": "balance", "steps": 90, "description": "Standing balance"},
        {"name": "preparation", "steps": 30, "description": "Movement preparation"},
        {"name": "walking", "steps": 300, "description": "Walking motion"},
        {"name": "stabilization", "steps": 60, "description": "Final stabilization"},
    ]

    total_steps = sum(phase["steps"] for phase in phases)
    print(
        f"📋 Recording plan: {len(phases)} phases, {total_steps} steps ({total_steps/60:.1f}s)"
    )

    try:
        # Start Genesis recording
        print("🔴 Starting Genesis camera recording...")
        camera.start_recording()

        current_step = 0

        for phase_idx, phase in enumerate(phases):
            print(
                f"\n📍 Phase {phase_idx + 1}: {phase['description']} ({phase['steps']} steps)"
            )

            for step in range(phase["steps"]):
                # Generate phase-appropriate robot actions
                t = current_step * (1 / 60)  # Time in seconds
                action = np.zeros(robot.n_dofs)

                if phase["name"] == "balance":
                    # Minimal movement for balance
                    action = action * 0.01

                elif phase["name"] == "preparation":
                    # Small preparatory movements
                    if robot.n_dofs > 6:
                        action[6] = 0.05 * np.sin(t * 2)  # Left hip
                    if robot.n_dofs > 7:
                        action[7] = -0.05 * np.sin(t * 2)  # Right hip

                elif phase["name"] == "walking":
                    # Full walking pattern
                    if robot.n_dofs >= 20:
                        # Hip movements
                        action[6] = 0.2 * np.sin(t * 3) + 0.05 * np.sin(
                            t * 9
                        )  # Left hip
                        action[7] = -0.2 * np.sin(t * 3) - 0.05 * np.sin(
                            t * 9
                        )  # Right hip

                        # Knee movements
                        action[12] = 0.15 * np.sin(t * 3 + np.pi / 2) + 0.1  # Left knee
                        action[13] = (
                            0.15 * np.sin(t * 3 - np.pi / 2) + 0.1
                        )  # Right knee

                        # Ankle movements
                        action[18] = 0.08 * np.sin(t * 3 + np.pi / 4)  # Left ankle
                        action[19] = -0.08 * np.sin(t * 3 - np.pi / 4)  # Right ankle

                        # Arm movements for balance
                        if robot.n_dofs > 25:
                            action[0] = 0.1 * np.sin(t * 3 + np.pi)  # Left arm
                            action[1] = -0.1 * np.sin(t * 3 + np.pi)  # Right arm

                elif phase["name"] == "stabilization":
                    # Gradual reduction of movement
                    decay = (phase["steps"] - step) / phase["steps"]
                    if robot.n_dofs > 6:
                        action[6] = 0.05 * np.sin(t) * decay
                        action[7] = -0.05 * np.sin(t) * decay

                # Apply robot control
                robot.set_dofs_kp([1000] * robot.n_dofs)
                robot.set_dofs_kv([100] * robot.n_dofs)
                robot.control_dofs_position(action)

                # Step simulation and render
                scene.step()
                camera.render(rgb=True)

                current_step += 1

                # Progress updates
                if step % 30 == 0:  # Every 0.5 seconds
                    progress = (current_step / total_steps) * 100
                    print(
                        f"  Progress: {progress:.1f}% (Step {current_step}/{total_steps})"
                    )

        print(f"\n✅ Recording completed: {total_steps} frames captured")

    except KeyboardInterrupt:
        print("\n⚠️ Recording interrupted by user")
    except Exception as e:
        print(f"\n❌ Recording error: {e}")

    finally:
        # Stop recording and save video
        try:
            print("💾 Saving video file...")
            camera.stop_recording(save_to_filename=str(video_path), fps=60)

            if video_path.exists():
                file_size = video_path.stat().st_size / (1024 * 1024)  # MB
                duration = total_steps / 60  # seconds

                print(f"✅ Video saved successfully!")
                print(f"   📁 File: {video_path}")
                print(f"   📏 Size: {file_size:.2f} MB")
                print(f"   ⏱️ Duration: {duration:.1f} seconds")
                print(f"   🎬 Resolution: 1280x720 @ 60 FPS")
                print(f"   🤖 Robot: Unitree G1 ({robot.n_dofs} DOF)")

                return str(video_path)
            else:
                print("❌ Video file was not created")
                return None

        except Exception as save_error:
            print(f"❌ Error saving video: {save_error}")
            return None


def main():
    """Main execution function."""
    print("🚀 Starting Genesis Camera Recording")
    print("This will create a high-quality video using Genesis rendering")
    print()

    start_time = time.time()
    video_path = record_genesis_video()
    elapsed = time.time() - start_time

    print(f"\n🎬 Recording session completed in {elapsed:.1f}s")

    if video_path:
        print(f"🎥 Video ready for playback:")
        print(f"   vlc {video_path}")
        print(f"   mpv {video_path}")
        print(f"   open {video_path}  # macOS")
    else:
        print("❌ Video recording failed")


if __name__ == "__main__":
    main()
