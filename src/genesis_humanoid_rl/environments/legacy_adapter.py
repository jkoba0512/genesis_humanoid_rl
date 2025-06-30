"""
Legacy adapter for backward compatibility.
Provides the old HumanoidWalkingEnv interface while using the new architecture.
"""

import warnings
import numpy as np
from gymnasium import Env
from typing import Any, Dict, Optional, Tuple

from .humanoid_env_v2 import (
    HumanoidWalkingEnvV2,
    EnvironmentConfig,
    make_humanoid_env_v2,
)
from ..di import create_container


class LegacyHumanoidWalkingEnv(Env):
    """
    Legacy adapter for HumanoidWalkingEnv.

    Maintains backward compatibility while using the new refactored architecture.
    Issues deprecation warnings to encourage migration to the new API.
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
        """
        Initialize legacy environment.

        Args:
            render_mode: Rendering mode
            simulation_fps: Physics simulation frequency
            control_freq: Control frequency
            episode_length: Maximum episode steps
            target_velocity: Target walking velocity
            **kwargs: Additional legacy parameters (ignored)
        """
        # Issue deprecation warning
        warnings.warn(
            "HumanoidWalkingEnv is deprecated. "
            "Use make_humanoid_env_v2() or HumanoidWalkingEnvV2 for new code. "
            "See migration guide in docs/migration.md",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert legacy parameters to new configuration
        config = EnvironmentConfig(
            simulation_fps=simulation_fps,
            control_freq=control_freq,
            max_episode_steps=episode_length,
            target_velocity=target_velocity,
            render_mode=render_mode,
        )

        # Create new environment with legacy configuration
        self._env = make_humanoid_env_v2(config=config)

        # Expose spaces for backward compatibility
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

        # Store legacy parameters
        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.control_freq = control_freq
        self.episode_length = episode_length
        self.target_velocity = target_velocity

        # Legacy state tracking
        self.current_step = 0

        # Warn about ignored kwargs
        if kwargs:
            ignored_params = list(kwargs.keys())
            warnings.warn(
                f"Legacy parameters ignored: {ignored_params}. "
                "Update to new API for full functionality.",
                UserWarning,
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        obs, info = self._env.reset(seed=seed, options=options)
        self.current_step = 0
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        return self._env.render()

    def close(self):
        """Clean up resources."""
        self._env.close()

    # Legacy properties for backward compatibility
    @property
    def scene(self):
        """Legacy scene property."""
        warnings.warn(
            "Direct scene access is deprecated. Use the new API.", DeprecationWarning
        )
        return getattr(self._env.physics, "scene", None)

    @property
    def robot(self):
        """Legacy robot property."""
        warnings.warn(
            "Direct robot access is deprecated. Use the new API.", DeprecationWarning
        )
        return getattr(self._env.physics, "robot", None)

    def _get_observation(self) -> np.ndarray:
        """Legacy observation method."""
        warnings.warn(
            "_get_observation is deprecated and no longer functional. "
            "Observations are handled automatically by the new API.",
            DeprecationWarning,
        )
        # Return current observation if available
        if hasattr(self._env, "robot_state") and self._env.robot_state is not None:
            from ..protocols import ObservationContext

            context = ObservationContext(
                previous_action=getattr(self._env, "previous_action", np.zeros(35)),
                target_velocity=self.target_velocity,
                step_count=self.current_step,
                additional_info={},
            )
            return self._env.observations.get_observation(
                self._env.robot_state, context
            )
        return np.zeros(self.observation_space.shape)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Legacy reward calculation method."""
        warnings.warn(
            "_calculate_reward is deprecated and no longer functional. "
            "Rewards are handled automatically by the new API.",
            DeprecationWarning,
        )
        return 0.0

    def _is_terminated(self) -> bool:
        """Legacy termination check method."""
        warnings.warn(
            "_is_terminated is deprecated and no longer functional. "
            "Termination is handled automatically by the new API.",
            DeprecationWarning,
        )
        return False


# Alias for complete backward compatibility
HumanoidWalkingEnv = LegacyHumanoidWalkingEnv


def create_migration_guide():
    """
    Print migration guide for users of the legacy API.
    """
    guide = """
    MIGRATION GUIDE: Legacy HumanoidWalkingEnv → New API
    
    OLD WAY:
    ```python
    from genesis_humanoid_rl.environments import HumanoidWalkingEnv
    
    env = HumanoidWalkingEnv(
        simulation_fps=100,
        control_freq=20,
        target_velocity=1.0
    )
    ```
    
    NEW WAY (Recommended):
    ```python
    from genesis_humanoid_rl.environments import make_humanoid_env_v2
    
    env = make_humanoid_env_v2(
        simulation_fps=100,
        control_freq=20,
        target_velocity=1.0
    )
    ```
    
    ADVANCED USAGE:
    ```python
    from genesis_humanoid_rl.environments import HumanoidWalkingEnvV2, EnvironmentConfig
    from genesis_humanoid_rl.di import create_container
    
    config = EnvironmentConfig(
        simulation_fps=100,
        control_freq=20,
        target_velocity=1.0
    )
    
    container = create_container()
    env = HumanoidWalkingEnvV2(config=config, container=container)
    ```
    
    BENEFITS OF NEW API:
    - Better testability and maintainability
    - Cleaner separation of concerns
    - Easy component swapping for research
    - Improved performance and error handling
    - Future-proof architecture
    
    BREAKING CHANGES:
    - Direct scene/robot access is deprecated
    - Internal methods (_get_observation, _calculate_reward) are deprecated
    - Some configuration parameters may have changed names
    
    For full details, see: docs/migration.md
    """
    print(guide)


def validate_legacy_usage():
    """
    Validate that legacy environment is being used correctly.

    Returns information about deprecated usage patterns.
    """
    import inspect
    import sys

    frame = inspect.currentframe()
    caller_frame = frame.f_back if frame else None

    if caller_frame:
        caller_filename = caller_frame.f_code.co_filename
        caller_lineno = caller_frame.f_lineno

        # Check if being imported from old location
        if "humanoid_env" in caller_filename:
            warnings.warn(
                f"Legacy import detected at {caller_filename}:{caller_lineno}. "
                "Consider updating to new API for better performance.",
                UserWarning,
            )

    return {
        "legacy_usage_detected": True,
        "recommendation": "Migrate to make_humanoid_env_v2() or HumanoidWalkingEnvV2",
        "migration_guide": "Run create_migration_guide() for detailed instructions",
    }
