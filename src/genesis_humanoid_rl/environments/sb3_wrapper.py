"""
Stable-Baselines3 compatible wrapper for the humanoid environment.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from typing import Any, Dict, Optional, Tuple

from .humanoid_env import HumanoidWalkingEnv


class SB3HumanoidEnv(gym.Env):
    """
    Stable-Baselines3 compatible wrapper for HumanoidWalkingEnv.

    This wrapper ensures full compatibility with SB3 training algorithms
    by handling any edge cases and providing proper Gymnasium interface.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        simulation_fps: int = 100,
        control_freq: int = 20,
        episode_length: int = 1000,
        target_velocity: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        # Create the base environment
        self.env = HumanoidWalkingEnv(
            render_mode=render_mode,
            simulation_fps=simulation_fps,
            control_freq=control_freq,
            episode_length=episode_length,
            target_velocity=target_velocity,
            **kwargs,
        )

        # Initialize environment to get spaces
        dummy_obs, _ = self.env.reset()

        # Set up spaces (SB3 requires these to be defined in __init__)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        # Store render mode
        self.render_mode = render_mode

        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)

        obs, info = self.env.reset()

        # Reset tracking
        self._episode_step = 0
        self._episode_reward = 0.0

        # Ensure observation is float32 (SB3 requirement)
        obs = obs.astype(np.float32)

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the environment."""
        # Ensure action is float32 and properly shaped
        action = np.array(action, dtype=np.float32)

        # Clip actions to be safe
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Update tracking
        self._episode_step += 1
        self._episode_reward += reward

        # Ensure observation is float32
        obs = obs.astype(np.float32)

        # Ensure reward is scalar float
        reward = float(reward)

        # Add episode statistics to info
        info.update(
            {
                "episode_step": self._episode_step,
                "episode_reward": self._episode_reward,
            }
        )

        # If episode is done, add final episode info
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_step,
            }

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        self.env.close()

    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
        return [seed]


def make_humanoid_env(
    env_id: str = "HumanoidWalk-v0", render_mode: Optional[str] = None, **env_kwargs
) -> SB3HumanoidEnv:
    """
    Factory function to create a SB3-compatible humanoid environment.

    Args:
        env_id: Environment identifier (for logging)
        render_mode: Render mode for visualization
        **env_kwargs: Additional environment arguments

    Returns:
        SB3HumanoidEnv: Ready-to-use environment
    """
    env = SB3HumanoidEnv(render_mode=render_mode, **env_kwargs)

    # Verify the environment is SB3-compatible
    try:
        check_env(env, warn=True)
        print("✓ Environment passed SB3 compatibility check")
    except Exception as e:
        print(f"⚠ Environment compatibility warning: {e}")

    return env


def test_environment():
    """Test the SB3 environment wrapper."""
    print("Testing SB3 humanoid environment...")

    # Create environment
    env = make_humanoid_env(
        episode_length=50, simulation_fps=100, control_freq=20  # Short for testing
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Test a few steps
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print(
            f"Step {step}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}"
        )

        if terminated or truncated:
            break

    print(f"Total reward: {total_reward:.3f}")
    env.close()
    print("✓ Environment test completed successfully")


if __name__ == "__main__":
    test_environment()
