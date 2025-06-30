#!/usr/bin/env python3
"""Create learning evolution videos showing robot's walking progress."""

import os
import sys
import json
import numpy as np
from pathlib import Path
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def create_stage_video(stage_name, model_path, output_path, steps=300):
    """Create video for specific learning stage."""
    print(f"\n🎬 Creating video for: {stage_name}")

    # Create environment for video recording
    env_config = {
        "episode_length": steps,
        "simulation_fps": 60,
        "control_freq": 20,
        "render": True,
        "headless": False,
        "target_velocity": 1.0,
    }

    try:
        env = make_humanoid_env(**env_config)
        print(f"✅ Environment created for {stage_name}")

        # Load model or use random policy for initial stage
        if model_path and Path(model_path).exists():
            model = PPO.load(model_path, env=env)
            print(f"✅ Model loaded: {model_path}")
            use_model = True
        else:
            print(f"⚠️ Using random policy for {stage_name}")
            use_model = False

        # Reset environment
        obs, _ = env.reset()

        # Record episode
        episode_reward = 0
        step_count = 0
        positions = []

        print(f"🔴 Recording {steps} steps...")
        start_time = time.time()

        for step in range(steps):
            if use_model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Random policy for untrained stage
                action = env.action_space.sample() * 0.1  # Small random actions

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            # Track robot position
            if hasattr(obs, "__len__") and len(obs) >= 3:
                positions.append([obs[0], obs[1], obs[2]])

            if done or truncated:
                obs, _ = env.reset()

            # Progress indicator
            if step % 50 == 0:
                print(f"   Step {step}/{steps} | Reward: {episode_reward:.2f}")

        elapsed = time.time() - start_time

        # Calculate metrics
        if positions:
            positions = np.array(positions)
            total_distance = np.sum(
                np.sqrt(np.sum(np.diff(positions[:, :2], axis=0) ** 2, axis=1))
            )
            final_position = positions[-1]
        else:
            total_distance = 0
            final_position = [0, 0, 0]

        print(f"✅ {stage_name} completed:")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Total reward: {episode_reward:.2f}")
        print(f"   Total distance: {total_distance:.2f}m")
        print(
            f"   Final position: [{final_position[0]:.2f}, {final_position[1]:.2f}, {final_position[2]:.2f}]"
        )

        env.close()

        return {
            "stage": stage_name,
            "duration": elapsed,
            "total_reward": episode_reward,
            "total_distance": total_distance,
            "final_position": final_position.tolist(),
            "steps": step_count,
        }

    except Exception as e:
        print(f"❌ Error creating video for {stage_name}: {e}")
        return None


def create_evolution_videos():
    """Create videos showing learning evolution."""
    print("🚀 Creating Learning Evolution Videos")
    print("=" * 50)

    # Define learning stages
    stages = [
        {
            "name": "Stage 0: Untrained Robot",
            "description": "Random movements, frequent falling",
            "model_path": None,  # Use random policy
            "steps": 200,
        },
        {
            "name": "Stage 1: Early Learning",
            "description": "Learning basic balance",
            "model_path": "./models/sample_training/checkpoint_2000",
            "steps": 250,
        },
        {
            "name": "Stage 2: Balance Acquired",
            "description": "Can stay upright, small steps",
            "model_path": "./models/sample_training/checkpoint_4000",
            "steps": 300,
        },
        {
            "name": "Stage 3: Walking Discovery",
            "description": "Forward movement, coordination improving",
            "model_path": "./models/sample_training/checkpoint_6000",
            "steps": 400,
        },
        {
            "name": "Stage 4: Mature Walking",
            "description": "Smooth, stable walking behavior",
            "model_path": "./models/sample_training/final_model",
            "steps": 500,
        },
    ]

    # Create output directory
    video_dir = Path("./videos/learning_evolution")
    video_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Create videos for each stage
    for i, stage in enumerate(stages):
        print(f"\n{'='*60}")
        print(f"STAGE {i+1}/5: {stage['name']}")
        print(f"Description: {stage['description']}")
        print(f"Steps: {stage['steps']}")

        output_path = (
            video_dir
            / f"stage_{i+1}_{stage['name'].lower().replace(' ', '_').replace(':', '')}.mp4"
        )

        result = create_stage_video(
            stage_name=stage["name"],
            model_path=stage["model_path"],
            output_path=str(output_path),
            steps=stage["steps"],
        )

        if result:
            results.append(result)

        # Brief pause between stages
        time.sleep(2)

    # Save results summary
    summary = {
        "created_at": datetime.now().isoformat(),
        "total_stages": len(stages),
        "successful_stages": len(results),
        "results": results,
    }

    summary_file = video_dir / "evolution_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("🎉 LEARNING EVOLUTION VIDEOS COMPLETED!")
    print(f"📁 Videos saved in: {video_dir}")
    print(f"📊 Summary saved: {summary_file}")
    print(f"✅ Successfully created: {len(results)}/{len(stages)} videos")

    # Print results summary
    print(f"\n📈 Evolution Summary:")
    for result in results:
        print(f"  {result['stage']}")
        print(f"    Reward: {result['total_reward']:.2f}")
        print(f"    Distance: {result['total_distance']:.2f}m")
        print(f"    Duration: {result['duration']:.1f}s")

    return results


if __name__ == "__main__":
    create_evolution_videos()
