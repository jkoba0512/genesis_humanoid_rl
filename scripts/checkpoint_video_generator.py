#!/usr/bin/env python3
"""Automated video generation for training checkpoints using Genesis camera."""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import time
import genesis as gs

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


class CheckpointVideoGenerator:
    """Generate high-quality evaluation videos at training checkpoints."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.video_dir = Path(config["monitoring"]["video_save_path"])
        self.video_dir.mkdir(parents=True, exist_ok=True)

        # Video settings
        self.video_length_seconds = 15  # 15 second videos
        self.fps = 60
        self.resolution = (1280, 720)

    def generate_all_checkpoint_videos(
        self, checkpoint_dir: str, phase_name: str
    ) -> List[str]:
        """Generate videos for all available checkpoints."""
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            print(f"❌ Checkpoint directory not found: {checkpoint_path}")
            return []

        # Find all checkpoint files
        checkpoint_files = list(checkpoint_path.glob("checkpoint_*.zip"))
        checkpoint_files.sort()

        if not checkpoint_files:
            print(f"❌ No checkpoint files found in {checkpoint_path}")
            return []

        print(f"🎬 Found {len(checkpoint_files)} checkpoints for video generation")

        video_paths = []
        for checkpoint_file in checkpoint_files:
            try:
                # Extract timesteps from filename
                timesteps = int(checkpoint_file.stem.split("_")[-1])

                video_path = self.create_checkpoint_video(
                    model_path=str(checkpoint_file),
                    timesteps=timesteps,
                    phase=phase_name,
                )

                if video_path:
                    video_paths.append(video_path)
                    print(f"✅ Video created: {Path(video_path).name}")

            except Exception as e:
                print(f"❌ Failed to create video for {checkpoint_file.name}: {e}")

        return video_paths

    def create_checkpoint_video(
        self, model_path: str, timesteps: int, phase: str
    ) -> Optional[str]:
        """Create high-quality Genesis video for a specific checkpoint."""
        print(f"🎬 Creating video for checkpoint {timesteps:,} timesteps...")

        try:
            # Initialize Genesis
            gs.init(backend=gs.cuda, logging_level="warning")

            # Create scene for video recording
            scene = gs.Scene(
                sim_options=gs.options.SimOptions(
                    dt=1 / 60,
                    substeps=4,
                ),
                viewer_options=gs.options.ViewerOptions(
                    res=self.resolution,
                    max_FPS=self.fps,
                    camera_pos=(3.0, -2.0, 1.5),
                    camera_lookat=(0.0, 0.0, 0.8),
                    camera_fov=40,
                ),
                vis_options=gs.options.VisOptions(
                    show_world_frame=False,
                    show_link_frame=False,
                    show_cameras=False,
                ),
                show_viewer=False,
            )

            # Add ground plane
            plane = scene.add_entity(gs.morphs.Plane())

            # Load G1 robot
            robot_path = Path("assets/robots/g1/g1_29dof.urdf")
            if not robot_path.exists():
                print(f"❌ Robot file not found: {robot_path}")
                return None

            robot = scene.add_entity(
                gs.morphs.URDF(
                    file=str(robot_path),
                    pos=(0, 0, 0.787),
                    euler=(0, 0, 0),
                ),
            )

            # Setup recording camera
            camera = scene.add_camera(
                res=self.resolution,
                pos=(3.5, -1.0, 1.2),  # Optimal viewing angle
                lookat=(0.0, 0.0, 0.8),
                fov=35,
                GUI=False,
            )

            # Build scene
            scene.build()

            # Load trained model for evaluation
            from stable_baselines3 import PPO

            model = PPO.load(model_path)

            # Create environment wrapper for observation
            env_config = self.config["environment"].copy()
            env_config["render"] = False
            env_config["headless"] = True
            env = make_humanoid_env(**env_config)

            # Prepare video output
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"{phase}_checkpoint_{timesteps}_{timestamp}.mp4"
            video_path = self.video_dir / video_filename

            print(f"🔴 Recording {self.video_length_seconds}s evaluation video...")

            # Start Genesis recording
            camera.start_recording()

            # Reset environment and get initial observation
            obs, _ = env.reset()

            # Performance tracking
            positions = []
            rewards = []
            total_reward = 0

            # Record evaluation
            total_frames = self.video_length_seconds * self.fps

            for frame in range(total_frames):
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Apply action to Genesis robot
                if robot.n_dofs == len(action):
                    robot.set_dofs_kp([1000] * robot.n_dofs)
                    robot.set_dofs_kv([100] * robot.n_dofs)
                    robot.control_dofs_position(action)

                # Step Genesis simulation
                scene.step()

                # Render frame for recording
                camera.render(rgb=True)

                # Step environment for observation
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward

                # Track position
                if len(obs) >= 3:
                    positions.append([obs[0], obs[1], obs[2]])
                rewards.append(reward)

                # Reset if episode ends
                if done or truncated:
                    obs, _ = env.reset()

                # Progress indicator
                if frame % (self.fps * 3) == 0:  # Every 3 seconds
                    progress = (frame / total_frames) * 100
                    print(f"  Recording progress: {progress:.1f}%")

            # Stop recording and save
            camera.stop_recording(save_to_filename=str(video_path), fps=self.fps)
            env.close()

            # Calculate performance metrics
            if positions:
                positions = np.array(positions)
                if len(positions) > 1:
                    distance_traveled = np.sqrt(
                        np.sum((positions[-1, :2] - positions[0, :2]) ** 2)
                    )
                    avg_height = np.mean(positions[:, 2])
                    max_distance = np.max(
                        [
                            np.sqrt(np.sum((pos[:2] - positions[0, :2]) ** 2))
                            for pos in positions
                        ]
                    )
                else:
                    distance_traveled = avg_height = max_distance = 0
            else:
                distance_traveled = avg_height = max_distance = 0

            # Save video metadata
            video_info = {
                "phase": phase,
                "timesteps": timesteps,
                "model_path": model_path,
                "video_path": str(video_path),
                "duration_seconds": self.video_length_seconds,
                "resolution": f"{self.resolution[0]}x{self.resolution[1]}",
                "fps": self.fps,
                "performance": {
                    "total_reward": float(total_reward),
                    "avg_reward": float(total_reward / len(rewards)) if rewards else 0,
                    "distance_traveled": float(distance_traveled),
                    "max_distance": float(max_distance),
                    "avg_height": float(avg_height),
                    "episode_count": (
                        int(np.sum([r > 0 for r in rewards])) if rewards else 0
                    ),
                    "success": distance_traveled > 3.0,  # 3m minimum for success
                },
                "timestamp": datetime.now().isoformat(),
            }

            # Save metadata
            info_path = video_path.with_suffix(".json")
            with open(info_path, "w") as f:
                json.dump(video_info, f, indent=2)

            # File size check
            if video_path.exists():
                file_size_mb = video_path.stat().st_size / (1024 * 1024)
                print(f"✅ Video saved: {video_path}")
                print(f"   📏 Size: {file_size_mb:.2f} MB")
                print(f"   📊 Distance: {distance_traveled:.2f}m")
                print(f"   🎯 Reward: {total_reward:.2f}")
                print(
                    f"   🏆 Success: {'✅' if video_info['performance']['success'] else '❌'}"
                )

                return str(video_path)
            else:
                print(f"❌ Video file was not created: {video_path}")
                return None

        except Exception as e:
            print(f"❌ Error creating video: {e}")
            return None

    def create_comparison_video(
        self, video_paths: List[str], phase: str
    ) -> Optional[str]:
        """Create a comparison video showing progress across checkpoints."""
        if len(video_paths) < 2:
            print("⚠️ Need at least 2 videos for comparison")
            return None

        print(f"🎬 Creating progress comparison video for {phase}...")

        # Load all video metadata
        video_infos = []
        for video_path in video_paths:
            info_path = Path(video_path).with_suffix(".json")
            if info_path.exists():
                with open(info_path, "r") as f:
                    video_infos.append(json.load(f))

        if not video_infos:
            print("❌ No video metadata found")
            return None

        # Sort by timesteps
        video_infos.sort(key=lambda x: x["timesteps"])

        # Create comparison summary
        comparison_data = {
            "phase": phase,
            "video_count": len(video_infos),
            "training_progression": [],
            "performance_summary": {
                "initial_distance": video_infos[0]["performance"]["distance_traveled"],
                "final_distance": video_infos[-1]["performance"]["distance_traveled"],
                "improvement": video_infos[-1]["performance"]["distance_traveled"]
                - video_infos[0]["performance"]["distance_traveled"],
                "initial_reward": video_infos[0]["performance"]["total_reward"],
                "final_reward": video_infos[-1]["performance"]["total_reward"],
            },
            "timestamp": datetime.now().isoformat(),
        }

        for info in video_infos:
            comparison_data["training_progression"].append(
                {
                    "timesteps": info["timesteps"],
                    "distance": info["performance"]["distance_traveled"],
                    "reward": info["performance"]["total_reward"],
                    "success": info["performance"]["success"],
                }
            )

        # Save comparison data
        comparison_path = self.video_dir / f"{phase}_training_progression.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f"📊 Training progression analysis saved: {comparison_path}")
        print(
            f"   📈 Distance improvement: {comparison_data['performance_summary']['improvement']:.2f}m"
        )
        print(
            f"   🎯 Final distance: {comparison_data['performance_summary']['final_distance']:.2f}m"
        )

        return str(comparison_path)


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate checkpoint evaluation videos"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to training configuration file"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory"
    )
    parser.add_argument("--phase", type=str, required=True, help="Training phase name")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = json.load(f)

    # Create video generator
    generator = CheckpointVideoGenerator(config)

    # Generate videos for all checkpoints
    video_paths = generator.generate_all_checkpoint_videos(
        checkpoint_dir=args.checkpoint_dir, phase_name=args.phase
    )

    if video_paths:
        print(f"\n✅ Generated {len(video_paths)} checkpoint videos")

        # Create comparison analysis
        comparison_path = generator.create_comparison_video(video_paths, args.phase)

        if comparison_path:
            print(f"📊 Training progression analysis: {comparison_path}")
    else:
        print("❌ No videos were generated")


if __name__ == "__main__":
    main()
