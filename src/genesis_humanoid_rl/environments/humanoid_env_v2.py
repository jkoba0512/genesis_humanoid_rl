"""
Refactored humanoid walking environment with proper separation of concerns.

This environment demonstrates clean architecture principles:
- Single Responsibility: Each manager handles one concern
- Dependency Injection: Dependencies are injected, not created
- Protocol-Based: Uses protocols for clean interfaces
- Testable: All components can be unit tested independently
"""

import numpy as np
from gymnasium import Env, spaces
from typing import Any, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from ..protocols import (
    PhysicsManagerProtocol,
    ObservationManagerProtocol,
    RewardCalculatorProtocol,
    TerminationCheckerProtocol,
    PhysicsConfig,
    ObservationContext,
    RobotState,
)
from ..di import create_container, DIContainer
from ..rewards.walking_rewards import WalkingRewardFunction

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for humanoid environment."""

    # Physics settings
    simulation_fps: int = 100
    control_freq: int = 20
    substeps: int = 10

    # Robot settings
    num_joints: int = 35
    robot_urdf_path: str = "assets/robots/g1/g1_29dof.urdf"
    initial_position: Tuple[float, float, float] = (0.0, 0.0, 0.8)

    # Episode settings
    max_episode_steps: int = 1000
    target_velocity: float = 1.0

    # Action settings
    action_scale: float = 0.1

    # Rendering
    render_mode: Optional[str] = None
    viewer_options: Optional[Dict[str, Any]] = None

    def get_physics_config(self) -> PhysicsConfig:
        """Get physics configuration."""
        return PhysicsConfig(
            simulation_fps=self.simulation_fps,
            control_freq=self.control_freq,
            substeps=self.substeps,
            render_mode=self.render_mode,
            viewer_options=self.viewer_options,
        )

    def get_initial_position(self) -> np.ndarray:
        """Get initial robot position as numpy array."""
        return np.array(self.initial_position)


class SimpleTerminationChecker:
    """Simple termination checker implementation."""

    def __init__(self):
        self._last_reason = None

    def should_terminate(self, robot_state: RobotState, config: Dict[str, Any]) -> bool:
        """Check if episode should terminate."""
        position = robot_state.position

        # Height termination - robot fell down
        if position[2] < 0.3:
            self._last_reason = "robot_fell"
            return True

        # Out of bounds termination
        if abs(position[0]) > 10.0 or abs(position[1]) > 5.0:
            self._last_reason = "out_of_bounds"
            return True

        # Excessive height termination (unrealistic jumping)
        if position[2] > 2.0:
            self._last_reason = "excessive_height"
            return True

        return False

    def get_termination_reason(self) -> Optional[str]:
        """Get reason for last termination."""
        return self._last_reason


class HumanoidWalkingEnvV2(Env):
    """
    Refactored humanoid walking environment with proper separation of concerns.

    This environment delegates responsibilities to specialized managers:
    - PhysicsManager: Handles Genesis simulation
    - ObservationManager: Generates observations
    - RewardCalculator: Computes rewards
    - TerminationChecker: Checks episode termination

    Benefits of this architecture:
    - Each component can be unit tested independently
    - Easy to swap implementations (e.g., different physics engines)
    - Clear separation of concerns
    - Better maintainability and extensibility
    """

    def __init__(
        self,
        physics_manager: Optional[PhysicsManagerProtocol] = None,
        observation_manager: Optional[ObservationManagerProtocol] = None,
        reward_calculator: Optional[RewardCalculatorProtocol] = None,
        termination_checker: Optional[TerminationCheckerProtocol] = None,
        config: Optional[EnvironmentConfig] = None,
        container: Optional[DIContainer] = None,
    ):
        """
        Initialize environment with dependency injection.

        Args:
            physics_manager: Physics simulation manager
            observation_manager: Observation generation manager
            reward_calculator: Reward calculation manager
            termination_checker: Termination condition checker
            config: Environment configuration
            container: DI container for automatic dependency resolution
        """
        super().__init__()

        # Use DI container if provided, otherwise create default
        if container is None:
            container = create_container()

        # Inject dependencies with fallbacks
        self.physics = physics_manager or container.get(PhysicsManagerProtocol)
        self.observations = observation_manager or container.get(
            ObservationManagerProtocol
        )
        self.rewards = reward_calculator or WalkingRewardFunction()
        self.termination_checker = termination_checker or SimpleTerminationChecker()

        # Configuration
        self.config = config or EnvironmentConfig()

        # Set up Gymnasium spaces
        self.action_space = self._create_action_space()
        self.observation_space = self.observations.get_observation_space()

        # Episode state
        self.current_step = 0
        self.previous_action = None
        self.robot_state = None

        # Configure physics manager
        if hasattr(self.physics, "set_action_scale"):
            self.physics.set_action_scale(self.config.action_scale)

        logger.info(
            f"Initialized HumanoidWalkingEnvV2 with {self.config.num_joints} joints"
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional reset options

        Returns:
            (initial_observation, info)
        """
        super().reset(seed=seed)

        try:
            # Reset all managers
            self.physics.reset_scene()
            self.observations.reset()
            self.rewards.reset()

            # Initialize physics scene
            physics_config = self.config.get_physics_config()
            self.physics.initialize_scene(physics_config)

            # Add robot to scene
            robot_urdf_path = self._get_absolute_urdf_path()
            initial_position = self.config.get_initial_position()
            self.physics.add_robot(robot_urdf_path, initial_position)

            # Get initial robot state
            self.robot_state = self.physics.get_robot_state()

            # Initialize episode state
            self.current_step = 0
            self.previous_action = np.zeros(self.config.num_joints)

            # Generate initial observation
            obs_context = ObservationContext(
                previous_action=self.previous_action,
                target_velocity=self.config.target_velocity,
                step_count=self.current_step,
                additional_info={},
            )

            observation = self.observations.get_observation(
                self.robot_state, obs_context
            )

            info = {
                "episode": "started",
                "step": self.current_step,
                "target_velocity": self.config.target_velocity,
            }

            logger.debug(
                f"Environment reset completed, observation shape: {observation.shape}"
            )
            return observation, info

        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            raise

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action vector (scaled to [-1, 1])

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.robot_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        try:
            # Validate and clip action
            action = np.clip(action, -1.0, 1.0)

            # Apply action through physics manager
            self.physics.apply_robot_control(action)

            # Step simulation
            steps_per_control = self.config.simulation_fps // self.config.control_freq
            self.physics.step_simulation(steps_per_control)

            # Get updated robot state
            self.robot_state = self.physics.get_robot_state()

            # Calculate reward
            total_reward, reward_components = self.rewards.compute_reward(
                self.robot_state.position,
                self.robot_state.orientation,
                self.robot_state.joint_positions,
                self.robot_state.joint_velocities,
                action,
            )

            # Update step counter
            self.current_step += 1

            # Check termination conditions
            terminated = self.termination_checker.should_terminate(self.robot_state, {})

            truncated = self.current_step >= self.config.max_episode_steps

            # Generate observation
            obs_context = ObservationContext(
                previous_action=self.previous_action,
                target_velocity=self.config.target_velocity,
                step_count=self.current_step,
                additional_info={},
            )

            observation = self.observations.get_observation(
                self.robot_state, obs_context
            )

            # Update state
            self.previous_action = action.copy()

            # Create info dictionary
            info = {
                "step": self.current_step,
                "terminated": terminated,
                "truncated": truncated,
                "reward_components": reward_components,
                "robot_position": self.robot_state.position.tolist(),
            }

            if terminated:
                info["termination_reason"] = (
                    self.termination_checker.get_termination_reason()
                )

            return observation, total_reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Failed to step environment: {e}")
            raise

    def render(self):
        """Render the environment."""
        if self.config.render_mode == "human":
            # Genesis handles rendering automatically when viewer is enabled
            pass
        elif self.config.render_mode == "rgb_array":
            # TODO: Implement RGB array rendering
            logger.warning("RGB array rendering not implemented")
            return None

    def close(self):
        """Clean up environment resources."""
        try:
            if self.physics:
                self.physics.close()
            logger.debug("Environment closed")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    def _create_action_space(self) -> spaces.Box:
        """Create action space definition."""
        return spaces.Box(
            low=-1.0, high=1.0, shape=(self.config.num_joints,), dtype=np.float32
        )

    def _get_absolute_urdf_path(self) -> str:
        """Get absolute path to robot URDF file."""
        import os

        # If path is already absolute, return as-is
        if os.path.isabs(self.config.robot_urdf_path):
            return self.config.robot_urdf_path

        # Otherwise, make it relative to project root
        current_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        )
        return os.path.join(current_dir, self.config.robot_urdf_path)

    def get_environment_info(self) -> Dict[str, Any]:
        """
        Get information about environment configuration and state.

        Returns:
            Dictionary with environment metadata
        """
        info = {
            "config": {
                "simulation_fps": self.config.simulation_fps,
                "control_freq": self.config.control_freq,
                "num_joints": self.config.num_joints,
                "max_episode_steps": self.config.max_episode_steps,
                "target_velocity": self.config.target_velocity,
                "action_scale": self.config.action_scale,
            },
            "spaces": {
                "action_space": str(self.action_space),
                "observation_space": str(self.observation_space),
            },
            "current_state": {
                "step": self.current_step,
                "robot_initialized": self.robot_state is not None,
            },
            "managers": {
                "physics": type(self.physics).__name__,
                "observations": type(self.observations).__name__,
                "rewards": type(self.rewards).__name__,
                "termination": type(self.termination_checker).__name__,
            },
        }

        if self.robot_state is not None:
            info["current_state"]["robot_position"] = self.robot_state.position.tolist()

        return info


# Factory function for easy environment creation
def make_humanoid_env_v2(
    config: Optional[EnvironmentConfig] = None,
    container: Optional[DIContainer] = None,
    **config_overrides,
) -> HumanoidWalkingEnvV2:
    """
    Factory function for creating HumanoidWalkingEnvV2.

    Args:
        config: Environment configuration
        container: DI container for dependency injection
        **config_overrides: Configuration overrides

    Returns:
        Configured environment instance
    """
    # Create or update configuration
    if config is None:
        config = EnvironmentConfig()

    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning(f"Unknown config parameter: {key}")

    # Create environment
    env = HumanoidWalkingEnvV2(config=config, container=container)

    logger.info(f"Created HumanoidWalkingEnvV2 with config: {config}")
    return env
