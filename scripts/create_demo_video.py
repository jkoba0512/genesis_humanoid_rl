#!/usr/bin/env python3
"""
Demo script for creating videos with Genesis cameras.

This script demonstrates the correct way to record videos with Genesis cameras,
including proper setup, recording, and common gotchas to avoid.
"""

import numpy as np
import genesis as gs
from pathlib import Path


def create_demo_video():
    """Create a demo video showing Genesis camera recording capabilities."""

    # Initialize Genesis with GPU backend for better performance
    gs.init(backend=gs.gpu)

    # Create scene with headless rendering (no GUI for video recording)
    scene = gs.Scene(
        show_viewer=False,  # Important: disable viewer for video recording
        renderer=gs.renderers.Rasterizer(),  # Use rasterizer for better performance
        rigid_options=gs.options.RigidOptions(
            dt=0.01,  # 100 FPS simulation
        ),
    )

    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # Add robot (using assets from the project)
    robot_path = (
        Path(__file__).parent.parent / "assets" / "robots" / "g1" / "g1_29dof.urdf"
    )
    if robot_path.exists():
        robot = scene.add_entity(
            gs.morphs.URDF(file=str(robot_path)),
            pos=(0, 0, 0.8),  # Start at walking height
        )
    else:
        # Fallback to a simple box if robot not available
        robot = scene.add_entity(
            gs.morphs.Box(size=(0.5, 0.3, 1.0)),
            pos=(0, 0, 0.8),
        )

    # Add multiple cameras for different angles
    cameras = []

    # Front camera (static)
    cam_front = scene.add_camera(
        res=(1280, 720),  # HD resolution
        pos=(3.0, 0.0, 1.0),
        lookat=(0, 0, 0.8),
        fov=45,
        GUI=False,  # Critical: must be False for recording
    )
    cameras.append(("front", cam_front))

    # Side camera (orbiting)
    cam_side = scene.add_camera(
        res=(1280, 720),
        pos=(0.0, 3.0, 1.0),
        lookat=(0, 0, 0.8),
        fov=45,
        GUI=False,
    )
    cameras.append(("side", cam_side))

    # Top-down camera (static)
    cam_top = scene.add_camera(
        res=(1280, 720),
        pos=(0.0, 0.0, 5.0),
        lookat=(0, 0, 0.8),
        fov=60,
        GUI=False,
    )
    cameras.append(("top", cam_top))

    # Build the scene
    scene.build()

    print("Starting video recording...")

    # Start recording on all cameras
    for name, cam in cameras:
        cam.start_recording()
        print(f"Recording started for {name} camera")

    # Simulation and recording loop
    n_frames = 300  # 3 seconds at 100 FPS

    for i in range(n_frames):
        # Step the simulation
        scene.step()

        # Move the orbiting camera
        angle = 2 * np.pi * i / n_frames
        orbit_radius = 3.0
        cam_side.set_pose(
            pos=(orbit_radius * np.cos(angle), orbit_radius * np.sin(angle), 1.5),
            lookat=(0, 0, 0.8),
        )

        # Apply some basic forces to the robot for movement
        if hasattr(robot, "set_dofs_kp"):
            # If it's a robot with joints, apply some simple walking motion
            t = i * 0.01  # simulation time
            joint_targets = np.sin(t * 2.0) * 0.3  # Simple sinusoidal motion
            robot.set_dofs_kp([100.0] * robot.n_dofs)
            robot.set_dofs_kd([10.0] * robot.n_dofs)
            robot.control_dofs_position([joint_targets] * robot.n_dofs)

        # Render all cameras
        for name, cam in cameras:
            cam.render()

        # Progress indicator
        if i % 60 == 0:
            print(f"Recording progress: {i/n_frames*100:.1f}%")

    print("Stopping recording and saving videos...")

    # Stop recording and save videos
    output_dir = Path(__file__).parent.parent / "videos"
    output_dir.mkdir(exist_ok=True)

    for name, cam in cameras:
        output_file = output_dir / f"demo_{name}.mp4"
        cam.stop_recording(save_to_filename=str(output_file), fps=60)
        print(f"Video saved: {output_file}")

    print("Demo video creation complete!")


def create_multi_modal_video():
    """Create a demo showing different rendering modes (RGB, depth, segmentation)."""

    gs.init(backend=gs.gpu)

    scene = gs.Scene(
        show_viewer=False,
        renderer=gs.renderers.Rasterizer(),
    )

    # Add some objects
    plane = scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=0.5),
        pos=(0, 0, 1.0),
        material=gs.materials.Rigid(color=(1.0, 0.0, 0.0)),
    )
    box = scene.add_entity(
        gs.morphs.Box(size=(0.3, 0.3, 0.3)),
        pos=(1.0, 0, 0.5),
        material=gs.materials.Rigid(color=(0.0, 1.0, 0.0)),
    )

    # Create camera
    cam = scene.add_camera(
        res=(640, 480),
        pos=(3.0, 3.0, 2.0),
        lookat=(0, 0, 0.5),
        fov=45,
        GUI=False,
    )

    scene.build()

    # Record with different rendering modes
    modes = [
        ("rgb", {}),
        ("depth", {"depth": True}),
        ("segmentation", {"segmentation": True}),
        ("normal", {"normal": True}),
    ]

    output_dir = Path(__file__).parent.parent / "videos"
    output_dir.mkdir(exist_ok=True)

    for mode_name, render_kwargs in modes:
        print(f"Recording {mode_name} video...")

        cam.start_recording()

        for i in range(120):  # 2 seconds
            scene.step()

            # Move objects
            sphere.set_pos((np.sin(i * 0.05), 0, 1.0 + 0.2 * np.cos(i * 0.1)))
            box.set_pos((1.0 + 0.3 * np.cos(i * 0.08), 0, 0.5))

            # Render with specific mode
            cam.render(**render_kwargs)

        output_file = output_dir / f"multimodal_{mode_name}.mp4"
        cam.stop_recording(save_to_filename=str(output_file), fps=60)
        print(f"Saved: {output_file}")


def main():
    """Main demo function."""
    print("Genesis Camera Video Recording Demo")
    print("===================================")

    try:
        # Create basic demo video
        create_demo_video()

        # Create multi-modal rendering demo
        create_multi_modal_video()

        print("\nAll demo videos created successfully!")
        print("Check the 'videos' directory for output files.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        print("Common issues:")
        print("1. Make sure Genesis is installed: pip install genesis-world")
        print("2. Ensure GPU drivers are up to date")
        print("3. Check that CUDA is available if using GPU backend")
        print("4. Verify robot assets exist in assets/robots/g1/")


if __name__ == "__main__":
    main()
