"""
Protocol definitions for clean interfaces between components.
Defines contracts for dependency injection and testing.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from gymnasium import spaces


@dataclass
class RobotState:
    """Complete robot state information."""

    position: np.ndarray  # (3,) - x, y, z base position
    orientation: np.ndarray  # (4,) - quaternion (x, y, z, w)
    joint_positions: np.ndarray  # (n_joints,) - joint positions
    joint_velocities: np.ndarray  # (n_joints,) - joint velocities
    timestamp: float = 0.0  # Simulation time


@dataclass
class PhysicsConfig:
    """Configuration for physics simulation."""

    simulation_fps: int = 100
    control_freq: int = 20
    substeps: int = 10
    render_mode: Optional[str] = None
    viewer_options: Optional[Dict[str, Any]] = None


@dataclass
class ObservationContext:
    """Context information for observation generation."""

    previous_action: np.ndarray
    target_velocity: float
    step_count: int
    additional_info: Dict[str, Any]


class PhysicsManagerProtocol(Protocol):
    """Protocol for physics simulation management."""

    def initialize_scene(self, config: PhysicsConfig) -> None:
        """Initialize physics scene with configuration."""
        ...

    def add_robot(self, urdf_path: str, position: np.ndarray) -> Any:
        """Add robot to scene and return robot entity."""
        ...

    def step_simulation(self, steps: int = 1) -> None:
        """Step the physics simulation."""
        ...

    def get_robot_state(self) -> RobotState:
        """Get current robot state."""
        ...

    def apply_robot_control(self, action: np.ndarray) -> None:
        """Apply control action to robot."""
        ...

    def reset_scene(self) -> None:
        """Reset the physics scene."""
        ...

    def close(self) -> None:
        """Clean up physics resources."""
        ...


class ObservationManagerProtocol(Protocol):
    """Protocol for observation generation."""

    def get_observation(
        self, robot_state: RobotState, context: ObservationContext
    ) -> np.ndarray:
        """Generate observation vector from robot state and context."""
        ...

    def get_observation_space(self) -> spaces.Box:
        """Get the observation space definition."""
        ...

    def reset(self) -> None:
        """Reset observation manager state."""
        ...


class RewardCalculatorProtocol(Protocol):
    """Protocol for reward calculation."""

    def compute_reward(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute reward for current state and action.

        Returns:
            (total_reward, reward_components)
        """
        ...

    def reset(self) -> None:
        """Reset reward calculator state."""
        ...

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update reward component weights."""
        ...


class TerminationCheckerProtocol(Protocol):
    """Protocol for termination condition checking."""

    def should_terminate(self, robot_state: RobotState, config: Dict[str, Any]) -> bool:
        """Check if episode should be terminated."""
        ...

    def get_termination_reason(self) -> Optional[str]:
        """Get reason for last termination."""
        ...


@dataclass
class EpisodeConfig:
    """Configuration for episode management."""

    max_steps: int = 1000
    termination_conditions: Dict[str, Any] = None

    def __post_init__(self):
        if self.termination_conditions is None:
            self.termination_conditions = {
                "min_height": 0.3,
                "max_height": 2.0,
                "max_x_distance": 10.0,
                "max_y_distance": 5.0,
            }


class EpisodeManagerProtocol(Protocol):
    """Protocol for episode lifecycle management."""

    def reset_episode(self) -> Dict[str, Any]:
        """Reset episode state and return initial info."""
        ...

    def step_episode(
        self, robot_state: RobotState, reward: float
    ) -> Tuple[bool, bool, Dict[str, Any]]:
        """
        Process episode step and check termination.

        Returns:
            (terminated, truncated, info)
        """
        ...

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode statistics."""
        ...


class EnvironmentConfigProtocol(Protocol):
    """Protocol for environment configuration."""

    def get_physics_config(self) -> PhysicsConfig:
        """Get physics configuration."""
        ...

    def get_episode_config(self) -> EpisodeConfig:
        """Get episode configuration."""
        ...

    def get_robot_urdf_path(self) -> str:
        """Get path to robot URDF file."""
        ...

    def get_initial_position(self) -> np.ndarray:
        """Get initial robot position."""
        ...


# Abstract base classes for common implementations


class BasePhysicsManager(ABC):
    """Base class for physics managers."""

    def __init__(self):
        self.scene = None
        self.robot = None
        self._initialized = False

    @abstractmethod
    def initialize_scene(self, config: PhysicsConfig) -> None:
        """Initialize physics scene."""
        pass

    @abstractmethod
    def add_robot(self, urdf_path: str, position: np.ndarray) -> Any:
        """Add robot to scene."""
        pass

    @abstractmethod
    def get_robot_state(self) -> RobotState:
        """Get robot state."""
        pass


class BaseObservationManager(ABC):
    """Base class for observation managers."""

    def __init__(self, num_joints: int = 35):
        self.num_joints = num_joints
        self._observation_space = self._create_observation_space()

    @abstractmethod
    def get_observation(
        self, robot_state: RobotState, context: ObservationContext
    ) -> np.ndarray:
        """Generate observation."""
        pass

    @abstractmethod
    def _create_observation_space(self) -> spaces.Box:
        """Create observation space."""
        pass

    def get_observation_space(self) -> spaces.Box:
        """Get observation space."""
        return self._observation_space


class BaseRewardCalculator(ABC):
    """Base class for reward calculators."""

    def __init__(self):
        self.weights = self._get_default_weights()
        self._previous_pos = None
        self._previous_action = None

    @abstractmethod
    def _get_default_weights(self) -> Dict[str, float]:
        """Get default reward weights."""
        pass

    @abstractmethod
    def compute_reward(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute reward."""
        pass

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update reward weights."""
        self.weights.update(weights)

    def reset(self) -> None:
        """Reset calculator state."""
        self._previous_pos = None
        self._previous_action = None


class BaseTerminationChecker(ABC):
    """Base class for termination checkers."""

    def __init__(self):
        self._last_termination_reason = None

    @abstractmethod
    def should_terminate(self, robot_state: RobotState, config: Dict[str, Any]) -> bool:
        """Check termination conditions."""
        pass

    def get_termination_reason(self) -> Optional[str]:
        """Get last termination reason."""
        return self._last_termination_reason
