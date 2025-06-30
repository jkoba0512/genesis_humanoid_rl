#!/usr/bin/env python3
"""Record actual video of robot walking using Genesis camera."""

import sys
import numpy as np
from pathlib import Path
import time
import genesis as gs

sys.path.insert(0, str(Path(__file__).parent.parent))


def record_walking_video():
    """Record actual video using Genesis camera system."""
    print("🎬 Recording Robot Walking Video")
    print("=" * 50)

    # Initialize Genesis with video recording
    gs.init(backend=gs.cuda, logging_level="warning")

    # Create scene for video recording
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            substeps=10,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.0, 0.0, 2.0),
            camera_lookat=(0.0, 0.0, 0.0),
            camera_fov=40,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
        ),
        show_viewer=False,  # Headless for video recording
    )

    # Add ground plane
    plane = scene.add_entity(gs.morphs.Plane())

    # Load G1 robot
    robot_path = Path("assets/robots/g1/g1_29dof.urdf")
    if not robot_path.exists():
        print(f"❌ Robot file not found: {robot_path}")
        return

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(robot_path),
            pos=(0, 0, 0.787),  # Grounded position
            euler=(0, 0, 0),
        ),
    )

    # Build scene
    print("Building scene for video recording...")
    scene.build()

    # Setup camera for recording
    cam = scene.add_camera(
        res=(1280, 720),
        pos=(2.0, 1.0, 1.5),
        lookat=(0, 0, 0.8),
        fov=50,
        GUI=False,  # For recording
    )

    print("🔴 Recording video...")

    frames = []
    n_steps = 300

    for step in range(n_steps):
        # Simple walking pattern
        t = step * 0.02
        action = np.zeros(robot.n_dofs)

        # Basic oscillating pattern for legs
        if robot.n_dofs >= 20:
            # Hip movements (assuming joints 6, 7 are left/right hip)
            if robot.n_dofs > 6:
                action[6] = 0.15 * np.sin(t)  # Left hip
            if robot.n_dofs > 7:
                action[7] = -0.15 * np.sin(t)  # Right hip

            # Knee movements (assuming joints 12, 13 are left/right knee)
            if robot.n_dofs > 12:
                action[12] = 0.1 * np.sin(t + np.pi / 2)  # Left knee
            if robot.n_dofs > 13:
                action[13] = 0.1 * np.sin(t - np.pi / 2)  # Right knee

            # Ankle movements
            if robot.n_dofs > 18:
                action[18] = 0.05 * np.sin(t)  # Left ankle
            if robot.n_dofs > 19:
                action[19] = -0.05 * np.sin(t)  # Right ankle

        # Apply action
        robot.set_dofs_kp([1000] * robot.n_dofs)
        robot.set_dofs_kv([100] * robot.n_dofs)
        robot.control_dofs_position(action)

        # Step simulation
        scene.step()

        # Capture frame
        rgba = cam.render(rgb=True, depth=False)
        frames.append(rgba[:, :, :3])  # Remove alpha channel

        if step % 50 == 0:
            print(f"  Recorded {step}/{n_steps} frames")

    print(f"✅ Captured {len(frames)} frames")

    # Save video
    output_path = Path("./videos/robot_walking.mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio

        print(f"💾 Saving video to {output_path}...")

        # Convert frames to proper format
        video_frames = []
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            video_frames.append(frame)

        # Save as MP4
        imageio.mimsave(
            str(output_path), video_frames, fps=30, quality=8, macro_block_size=1
        )

        print(f"✅ Video saved: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"   Duration: {len(frames) / 30:.1f} seconds")
        print(f"   Resolution: 1280x720")
        print(f"   FPS: 30")

        return str(output_path)

    except ImportError:
        print("❌ imageio not available. Installing...")
        import subprocess

        subprocess.run(["pip", "install", "imageio[ffmpeg]"], check=True)

        import imageio

        imageio.mimsave(str(output_path), frames, fps=30)
        return str(output_path)

    except Exception as e:
        print(f"❌ Error saving video: {e}")

        # Fallback: save individual frames
        frame_dir = Path("./videos/frames")
        frame_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_path = frame_dir / f"frame_{i:04d}.png"
            imageio.imwrite(str(frame_path), frame)

        print(f"💾 Saved {len(frames)} frames to {frame_dir}")
        print(
            "   Use: ffmpeg -r 30 -i frames/frame_%04d.png -c:v libx264 robot_walking.mp4"
        )

        return str(frame_dir)


if __name__ == "__main__":
    video_path = record_walking_video()
    print(f"\n🎬 Video recording completed!")
    print(f"📁 Output: {video_path}")
