"""
Physics management abstraction for Genesis engine.
Handles all Genesis-specific operations in isolation.
"""

import numpy as np
import torch
import genesis as gs
import logging
from typing import Any, Optional, Dict
import time

from ..protocols import (
    PhysicsManagerProtocol,
    PhysicsConfig,
    RobotState,
    BasePhysicsManager,
)
from .robot_grounding import RobotGroundingFactory

logger = logging.getLogger(__name__)


class GenesisPhysicsManager(BasePhysicsManager):
    """
    Genesis-specific physics management.

    Abstracts all Genesis operations to enable:
    - Clean unit testing with mocks
    - Potential engine swapping in the future
    - Clear separation of physics from environment logic
    """

    def __init__(self, robot_grounding_factory=None):
        super().__init__()
        self.robot_grounding_factory = (
            robot_grounding_factory or RobotGroundingFactory()
        )
        self._current_action_scale = 0.1
        self._control_frequency = 20
        self._initialized_genesis = False

    def initialize_scene(self, config: PhysicsConfig) -> None:
        """
        Initialize Genesis scene with configuration.

        Args:
            config: Physics configuration parameters
        """
        try:
            # Initialize Genesis only if not already done
            if not self._initialized_genesis:
                gs.init()
                self._initialized_genesis = True
                logger.info("Genesis physics engine initialized")
        except RuntimeError as e:
            if "Genesis already initialized" not in str(e):
                raise
            logger.debug("Genesis already initialized")

        # Configure scene options
        scene_kwargs = {
            "sim_options": gs.options.SimOptions(
                dt=1.0 / config.simulation_fps,
                substeps=config.substeps,
            ),
            "show_viewer": config.render_mode is not None,
        }

        # Add viewer options if rendering
        if config.render_mode is not None and config.viewer_options:
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                **config.viewer_options
            )

        # Create scene
        self.scene = gs.Scene(**scene_kwargs)

        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Store configuration
        self._control_frequency = config.control_freq

        logger.info(f"Genesis scene initialized with {config.simulation_fps} FPS")

    def add_robot(self, urdf_path: str, position: np.ndarray) -> Any:
        """
        Add robot to physics scene.

        Args:
            urdf_path: Path to robot URDF file
            position: Initial position (x, y, z)

        Returns:
            Robot entity reference
        """
        if self.scene is None:
            raise RuntimeError("Scene not initialized. Call initialize_scene() first.")

        # Load robot at initial position
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=tuple(position),
                euler=(0, 0, 0),
            )
        )

        # Build scene (required before robot operations)
        self.scene.build()

        # Apply automatic grounding
        if self.robot is not None:
            try:
                grounding_calculator = self.robot_grounding_factory.create_calculator(
                    self.robot, verbose=True
                )
                grounding_height = grounding_calculator.get_grounding_height(
                    safety_margin=0.03
                )
                self.robot.set_pos(
                    torch.tensor([position[0], position[1], grounding_height])
                )

                # Let robot settle
                for _ in range(10):
                    self.scene.step()

                logger.info(f"Robot positioned at height {grounding_height:.3f}m")

            except Exception as e:
                logger.warning(f"Robot grounding failed: {e}. Using default position.")

        logger.info(
            f"Robot loaded with {self.robot.n_dofs} DOFs and {self.robot.n_links} links"
        )
        return self.robot

    def step_simulation(self, steps: int = 1) -> None:
        """
        Step the physics simulation.

        Args:
            steps: Number of simulation steps to execute
        """
        if self.scene is None:
            raise RuntimeError("Scene not initialized")

        for _ in range(steps):
            self.scene.step()

    def get_robot_state(self) -> RobotState:
        """
        Get current robot state.

        Returns:
            Complete robot state information
        """
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        # Extract state from Genesis robot
        position = self.robot.get_pos().cpu().numpy()
        orientation = self.robot.get_quat().cpu().numpy()
        joint_positions = self.robot.get_dofs_position().cpu().numpy()
        joint_velocities = self.robot.get_dofs_velocity().cpu().numpy()

        return RobotState(
            position=position,
            orientation=orientation,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            timestamp=time.time(),
        )

    def apply_robot_control(self, action: np.ndarray) -> None:
        """
        Apply control action to robot.

        Args:
            action: Control action vector (scaled to [-1, 1])
        """
        if self.robot is None:
            raise RuntimeError("Robot not initialized")

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Get current joint positions
        current_pos = self.robot.get_dofs_position()

        # Convert action to position deltas
        # Scale from [-1, 1] to reasonable position changes
        position_delta = torch.tensor(
            action * self._current_action_scale,
            dtype=current_pos.dtype,
            device=current_pos.device,
        )

        # Compute target positions
        target_pos = current_pos + position_delta

        # Apply position control
        self.robot.control_dofs_position(target_pos)

    def reset_scene(self) -> None:
        """Reset the physics scene to initial state."""
        # For Genesis, we typically create a new scene for reset
        # This could be optimized with scene pooling in the future
        self.scene = None
        self.robot = None
        logger.debug("Physics scene reset")

    def close(self) -> None:
        """Clean up physics resources."""
        # Genesis handles cleanup automatically
        self.scene = None
        self.robot = None
        logger.debug("Physics manager closed")

    def set_action_scale(self, scale: float) -> None:
        """
        Set the action scaling factor.

        Args:
            scale: Scaling factor for action to position conversion
        """
        self._current_action_scale = scale
        logger.debug(f"Action scale set to {scale}")

    def get_simulation_info(self) -> Dict[str, Any]:
        """
        Get simulation information and statistics.

        Returns:
            Dictionary with simulation metadata
        """
        info = {
            "scene_initialized": self.scene is not None,
            "robot_loaded": self.robot is not None,
            "action_scale": self._current_action_scale,
            "control_frequency": self._control_frequency,
        }

        if self.robot is not None:
            info.update(
                {
                    "robot_dofs": self.robot.n_dofs,
                    "robot_links": self.robot.n_links,
                }
            )

        return info


class MockPhysicsManager(BasePhysicsManager):
    """
    Mock physics manager for testing.

    Provides the same interface as GenesisPhysicsManager but without
    actual physics simulation, enabling fast unit tests.
    """

    def __init__(self):
        super().__init__()
        self._step_count = 0
        self._robot_position = np.array([0.0, 0.0, 0.8])
        self._robot_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self._joint_positions = np.zeros(35)
        self._joint_velocities = np.zeros(35)

    def initialize_scene(self, config: PhysicsConfig) -> None:
        """Mock scene initialization."""
        self._initialized = True
        logger.debug("Mock physics scene initialized")

    def add_robot(self, urdf_path: str, position: np.ndarray) -> Any:
        """Mock robot addition."""
        self._robot_position = position.copy()

        # Create mock robot object
        class MockRobot:
            n_dofs = 35
            n_links = 30

        self.robot = MockRobot()
        logger.debug("Mock robot added")
        return self.robot

    def step_simulation(self, steps: int = 1) -> None:
        """Mock simulation stepping."""
        self._step_count += steps

        # Simulate slight forward movement
        self._robot_position[0] += 0.001 * steps

    def get_robot_state(self) -> RobotState:
        """Mock robot state retrieval."""
        return RobotState(
            position=self._robot_position.copy(),
            orientation=self._robot_orientation.copy(),
            joint_positions=self._joint_positions.copy(),
            joint_velocities=self._joint_velocities.copy(),
            timestamp=self._step_count * 0.01,  # Mock time
        )

    def apply_robot_control(self, action: np.ndarray) -> None:
        """Mock robot control."""
        # Update joint positions based on action
        self._joint_positions += action * 0.01

    def reset_scene(self) -> None:
        """Mock scene reset."""
        self._step_count = 0
        self._robot_position = np.array([0.0, 0.0, 0.8])
        self._joint_positions = np.zeros(35)
        self._joint_velocities = np.zeros(35)

    def close(self) -> None:
        """Mock cleanup."""
        pass
