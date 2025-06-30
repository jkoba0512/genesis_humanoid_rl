"""
Observation management for humanoid environments.
Handles the construction of observation vectors from robot state.
"""

import numpy as np
from gymnasium import spaces
from typing import Dict, Any
import logging

from ..protocols import (
    ObservationManagerProtocol,
    ObservationContext,
    RobotState,
    BaseObservationManager,
)

logger = logging.getLogger(__name__)


class HumanoidObservationManager(BaseObservationManager):
    """
    Observation manager for humanoid robots.

    Constructs observation vectors from robot state and context information.
    The observation includes:
    - Base position (3D)
    - Base orientation (quaternion, 4D)
    - Joint positions (n_joints)
    - Joint velocities (n_joints)
    - Previous action (n_joints)
    - Target velocity (1D)
    """

    def __init__(self, num_joints: int = 35, normalize: bool = False):
        """
        Initialize observation manager.

        Args:
            num_joints: Number of robot joints
            normalize: Whether to apply normalization to observations
        """
        self.num_joints = num_joints
        self.normalize = normalize
        self._observation_ranges = self._get_observation_ranges()
        super().__init__(num_joints)

        logger.info(f"Initialized observation manager for {num_joints} joints")

    def get_observation(
        self, robot_state: RobotState, context: ObservationContext
    ) -> np.ndarray:
        """
        Generate observation vector from robot state and context.

        Args:
            robot_state: Current robot state
            context: Additional context information

        Returns:
            Observation vector as numpy array
        """
        # Validate inputs
        self._validate_robot_state(robot_state)
        self._validate_context(context)

        # Construct observation components
        obs_components = [
            robot_state.position,  # Base position (3)
            robot_state.orientation,  # Base orientation (4)
            robot_state.joint_positions,  # Joint positions (n_joints)
            robot_state.joint_velocities,  # Joint velocities (n_joints)
            context.previous_action,  # Previous action (n_joints)
            np.array([context.target_velocity]),  # Target velocity (1)
        ]

        # Concatenate all components
        observation = np.concatenate(obs_components)

        # Apply normalization if enabled
        if self.normalize:
            observation = self._normalize_observation(observation)

        # Ensure correct dtype
        observation = observation.astype(np.float32)

        # Validate observation
        self._validate_observation(observation)

        return observation

    def _create_observation_space(self) -> spaces.Box:
        """
        Create observation space definition.

        Returns:
            Gymnasium Box space for observations
        """
        # Calculate observation dimension
        # Position (3) + orientation (4) + joint_pos (n_joints) +
        # joint_vel (n_joints) + prev_action (n_joints) + target_vel (1)
        obs_dim = 3 + 4 + self.num_joints + self.num_joints + self.num_joints + 1

        if self.normalize:
            # Normalized observations are in [-1, 1] range
            low = -1.0
            high = 1.0
        else:
            # Use infinite bounds for unnormalized observations
            low = -np.inf
            high = np.inf

        return spaces.Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)

    def _get_observation_ranges(self) -> Dict[str, tuple]:
        """
        Get expected ranges for different observation components.
        Used for normalization and validation.
        """
        return {
            "position": (-10.0, 10.0),  # Robot position in meters
            "orientation": (-1.0, 1.0),  # Quaternion components
            "joint_positions": (-np.pi, np.pi),  # Joint positions in radians
            "joint_velocities": (-10.0, 10.0),  # Joint velocities in rad/s
            "action": (-1.0, 1.0),  # Previous action (normalized)
            "target_velocity": (0.0, 5.0),  # Target velocity in m/s
        }

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation components to [-1, 1] range.

        Args:
            observation: Raw observation vector

        Returns:
            Normalized observation vector
        """
        normalized = observation.copy()

        # Define component indices
        pos_start, pos_end = 0, 3
        quat_start, quat_end = 3, 7
        joint_pos_start = 7
        joint_pos_end = 7 + self.num_joints
        joint_vel_start = joint_pos_end
        joint_vel_end = joint_vel_start + self.num_joints
        action_start = joint_vel_end
        action_end = action_start + self.num_joints
        target_vel_idx = action_end

        ranges = self._observation_ranges

        # Normalize position
        pos_min, pos_max = ranges["position"]
        normalized[pos_start:pos_end] = (
            2.0 * ((observation[pos_start:pos_end] - pos_min) / (pos_max - pos_min))
            - 1.0
        )

        # Orientation (quaternion) is already in [-1, 1]
        # No normalization needed for quaternion

        # Normalize joint positions
        joint_min, joint_max = ranges["joint_positions"]
        normalized[joint_pos_start:joint_pos_end] = (
            2.0
            * (
                (observation[joint_pos_start:joint_pos_end] - joint_min)
                / (joint_max - joint_min)
            )
            - 1.0
        )

        # Normalize joint velocities
        vel_min, vel_max = ranges["joint_velocities"]
        normalized[joint_vel_start:joint_vel_end] = (
            2.0
            * (
                (observation[joint_vel_start:joint_vel_end] - vel_min)
                / (vel_max - vel_min)
            )
            - 1.0
        )

        # Previous action is already normalized
        # No normalization needed

        # Normalize target velocity
        target_min, target_max = ranges["target_velocity"]
        target_vel = observation[target_vel_idx]
        normalized[target_vel_idx] = (
            2.0 * ((target_vel - target_min) / (target_max - target_min)) - 1.0
        )

        # Clip to ensure values stay in [-1, 1]
        normalized = np.clip(normalized, -1.0, 1.0)

        return normalized

    def _validate_robot_state(self, robot_state: RobotState) -> None:
        """Validate robot state structure."""
        if not isinstance(robot_state, RobotState):
            raise TypeError(f"Expected RobotState, got {type(robot_state)}")

        # Check array shapes
        if robot_state.position.shape != (3,):
            raise ValueError(f"Invalid position shape: {robot_state.position.shape}")

        if robot_state.orientation.shape != (4,):
            raise ValueError(
                f"Invalid orientation shape: {robot_state.orientation.shape}"
            )

        if robot_state.joint_positions.shape != (self.num_joints,):
            raise ValueError(
                f"Invalid joint positions shape: {robot_state.joint_positions.shape}"
            )

        if robot_state.joint_velocities.shape != (self.num_joints,):
            raise ValueError(
                f"Invalid joint velocities shape: {robot_state.joint_velocities.shape}"
            )

    def _validate_context(self, context: ObservationContext) -> None:
        """Validate observation context."""
        if not isinstance(context, ObservationContext):
            raise TypeError(f"Expected ObservationContext, got {type(context)}")

        if context.previous_action.shape != (self.num_joints,):
            raise ValueError(
                f"Invalid previous action shape: {context.previous_action.shape}"
            )

        if not isinstance(context.target_velocity, (int, float)):
            raise TypeError(
                f"Target velocity must be numeric, got {type(context.target_velocity)}"
            )

    def _validate_observation(self, observation: np.ndarray) -> None:
        """Validate final observation."""
        expected_shape = self.get_observation_space().shape

        if observation.shape != expected_shape:
            raise ValueError(
                f"Invalid observation shape: {observation.shape}, expected {expected_shape}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError("Observation contains invalid values (NaN or inf)")

    def reset(self) -> None:
        """Reset observation manager state."""
        # No internal state to reset for this implementation
        logger.debug("Observation manager reset")

    def get_observation_info(self) -> Dict[str, Any]:
        """
        Get information about observation structure.

        Returns:
            Dictionary with observation metadata
        """
        return {
            "num_joints": self.num_joints,
            "observation_dim": self.get_observation_space().shape[0],
            "normalized": self.normalize,
            "components": {
                "position": (0, 3),
                "orientation": (3, 7),
                "joint_positions": (7, 7 + self.num_joints),
                "joint_velocities": (7 + self.num_joints, 7 + 2 * self.num_joints),
                "previous_action": (7 + 2 * self.num_joints, 7 + 3 * self.num_joints),
                "target_velocity": (
                    7 + 3 * self.num_joints,
                    7 + 3 * self.num_joints + 1,
                ),
            },
        }


class MockObservationManager(BaseObservationManager):
    """
    Mock observation manager for testing.

    Provides the same interface but generates simple synthetic observations
    for fast unit testing without requiring actual robot state.
    """

    def __init__(self, num_joints: int = 35):
        super().__init__(num_joints)
        self._call_count = 0

    def get_observation(
        self, robot_state: RobotState, context: ObservationContext
    ) -> np.ndarray:
        """Generate mock observation."""
        self._call_count += 1

        # Generate deterministic but varying observation
        obs_dim = self.get_observation_space().shape[0]
        observation = np.zeros(obs_dim, dtype=np.float32)

        # Add some variation based on call count
        variation = 0.1 * np.sin(self._call_count * 0.1)

        # Set base position
        observation[0] = variation  # x position varies
        observation[1] = 0.0  # y position constant
        observation[2] = 0.8  # z position constant

        # Set orientation (identity quaternion)
        observation[3:7] = [0.0, 0.0, 0.0, 1.0]

        # Set joint positions (small variation)
        joint_start = 7
        observation[joint_start : joint_start + self.num_joints] = variation * 0.1

        # Joint velocities (zero)
        vel_start = joint_start + self.num_joints
        observation[vel_start : vel_start + self.num_joints] = 0.0

        # Previous action (zero)
        action_start = vel_start + self.num_joints
        observation[action_start : action_start + self.num_joints] = 0.0

        # Target velocity
        observation[-1] = context.target_velocity

        return observation

    def _create_observation_space(self) -> spaces.Box:
        """Create mock observation space."""
        obs_dim = 3 + 4 + self.num_joints + self.num_joints + self.num_joints + 1
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def reset(self) -> None:
        """Reset mock observation manager."""
        self._call_count = 0
