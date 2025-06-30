#!/usr/bin/env python3
"""
Debug the termination issue in the reward function.
"""

import numpy as np
import os
import sys

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from genesis_humanoid_rl.environments.humanoid_env import HumanoidWalkingEnv


def debug_termination():
    """Debug why episodes are terminating immediately."""
    print("=== Debugging Termination Issue ===")

    # Create environment
    env = HumanoidWalkingEnv(
        render_mode=None,  # Headless for testing
        simulation_fps=100,
        control_freq=20,
        episode_length=50,  # Very short for debugging
        target_velocity=1.0,
    )

    print("Environment created, calling reset...")
    obs, info = env.reset()
    print(f"Reset completed. Observation shape: {obs.shape}")

    # Check initial robot state
    if env.robot is not None:
        pos = env.robot.get_pos().cpu().numpy()
        quat = env.robot.get_quat().cpu().numpy()

        print(f"Initial robot position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(
            f"Initial robot quaternion: [{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]"
        )

        # Test termination conditions manually
        print("\n=== Checking Termination Conditions ===")

        # 1. Height check
        height_ok = pos[2] >= 0.3
        print(f"Height check (pos[2] >= 0.3): {pos[2]:.3f} >= 0.3 = {height_ok}")

        # 2. Uprightness check
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            rotation_magnitude = np.sqrt(quat[0] ** 2 + quat[1] ** 2)
            upright_ok = rotation_magnitude <= 0.8
            print(
                f"Uprightness check: rotation_magnitude={rotation_magnitude:.3f} <= 0.8 = {upright_ok}"
            )
        else:
            upright_ok = False
            print(f"Uprightness check: quaternion norm is 0, upright_ok = {upright_ok}")

        # 3. Bounds check
        bounds_ok = abs(pos[0]) <= 10.0 and abs(pos[1]) <= 5.0
        print(
            f"Bounds check: |{pos[0]:.3f}| <= 10.0 and |{pos[1]:.3f}| <= 5.0 = {bounds_ok}"
        )

        # 4. Height ceiling check
        ceiling_ok = pos[2] <= 2.0
        print(f"Height ceiling check: {pos[2]:.3f} <= 2.0 = {ceiling_ok}")

        # Overall termination
        should_terminate = env._is_terminated()
        print(f"\nOverall termination result: {should_terminate}")

        if should_terminate:
            print(
                "❌ Robot should terminate immediately after reset - this is the problem!"
            )
        else:
            print("✅ Robot should not terminate - termination logic is correct")

    # Test a few steps
    print(f"\n=== Testing Steps ===")
    for step in range(5):
        action = np.zeros(env.action_space.shape)  # No movement
        obs, reward, terminated, truncated, info = env.step(action)

        if env.robot is not None:
            pos = env.robot.get_pos().cpu().numpy()

        print(
            f"Step {step}: reward={reward:.3f}, terminated={terminated}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        )

        if terminated:
            print(f"  ❌ Episode terminated at step {step}")
            break

    print("\n=== Diagnosis Complete ===")


if __name__ == "__main__":
    debug_termination()
