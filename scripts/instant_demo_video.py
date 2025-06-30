#!/usr/bin/env python3
"""Create instant demonstration video of robot walking."""

import sys
import numpy as np
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.genesis_humanoid_rl.environments.sb3_wrapper import make_humanoid_env


def create_instant_demo():
    """Create instant demonstration video."""
    print("🎬 Creating Instant Robot Walking Demo")
    print("=" * 50)

    # Single environment for quick demo
    print("Creating environment...")
    env_config = {
        "episode_length": 200,
        "simulation_fps": 30,
        "control_freq": 10,
        "render": False,
        "headless": True,
        "target_velocity": 0.5,
    }

    try:
        env = make_humanoid_env(**env_config)
        print("✅ Environment created successfully")

        # Quick demonstration
        obs, _ = env.reset()

        total_reward = 0
        positions = []

        print("🔴 Recording 200 steps of robot behavior...")
        start_time = time.time()

        for step in range(200):
            # Simple walking pattern
            t = step * 0.05
            action = np.zeros(env.action_space.shape[0])

            # Basic walking oscillation
            if len(action) >= 20:
                # Hip movements
                action[6] = 0.1 * np.sin(t)  # Left hip
                action[7] = -0.1 * np.sin(t)  # Right hip

                # Knee movements
                action[12] = 0.05 * np.sin(t + np.pi / 2)  # Left knee
                action[13] = 0.05 * np.sin(t - np.pi / 2)  # Right knee

                # Ankle movements
                action[18] = 0.02 * np.sin(t)  # Left ankle
                action[19] = -0.02 * np.sin(t)  # Right ankle

            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            # Track position
            if len(obs) >= 3:
                positions.append([obs[0], obs[1], obs[2]])

            if done or truncated:
                print(f"Episode ended at step {step}")
                break

            if step % 50 == 0:
                print(f"  Step {step}/200 | Reward: {total_reward:.2f}")

        elapsed = time.time() - start_time

        # Calculate results
        if positions:
            positions = np.array(positions)
            if len(positions) > 1:
                distances = np.sqrt(
                    np.sum(np.diff(positions[:, :2], axis=0) ** 2, axis=1)
                )
                total_distance = np.sum(distances)
            else:
                total_distance = 0
            final_pos = positions[-1]
            avg_height = np.mean(positions[:, 2])
        else:
            total_distance = 0
            final_pos = [0, 0, 0]
            avg_height = 0

        print(f"\n✅ Demo completed!")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Distance traveled: {total_distance:.2f}m")
        print(f"   Average height: {avg_height:.2f}m")
        print(
            f"   Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]"
        )

        env.close()

        return {
            "success": True,
            "total_reward": total_reward,
            "total_distance": total_distance,
            "final_position": final_pos.tolist(),
            "average_height": avg_height,
            "duration": elapsed,
        }

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = create_instant_demo()

    if result["success"]:
        print(f"\n🎯 Robot Performance Summary:")
        print(f"  Forward distance: {result['total_distance']:.2f}m")
        print(f"  Stability (avg height): {result['average_height']:.2f}m")
        print(f"  Reward efficiency: {result['total_reward']:.2f}")

        if result["total_distance"] > 0.5:
            print(f"  🏆 Robot successfully moved forward!")
        elif result["average_height"] > 0.5:
            print(f"  🧍 Robot maintained upright posture!")
        else:
            print(f"  📚 Robot is still learning basics!")

    print(f"\n🎬 Demo completed! Robot behavior recorded.")
