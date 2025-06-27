"""
Simulation and environment test fixtures.

Provides mock Genesis environments, robot states, and physics simulation
components for testing without real physics dependencies.
"""

import pytest
import numpy as np
import torch
from contextlib import contextmanager
from typing import Dict, Any, Optional, Generator, List
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from src.genesis_humanoid_rl.protocols import RobotState


@dataclass
class MockRobotState:
    """Enhanced mock robot state with realistic physics simulation."""
    position: np.ndarray
    orientation: np.ndarray  # quaternion [x, y, z, w]
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    contact_forces: Optional[np.ndarray] = None
    step_count: int = 0
    simulation_time: float = 0.0
    
    @property
    def n_dofs(self) -> int:
        """Number of degrees of freedom."""
        return len(self.joint_positions)
    
    @property
    def n_links(self) -> int:
        """Number of robot links."""
        return 30  # Unitree G1 default


class MockPhysicsEngine:
    """
    Mock physics engine that simulates realistic robot behavior.
    
    Provides deterministic, controllable physics simulation for testing
    without requiring actual Genesis installation.
    """
    
    def __init__(self, initial_state: Optional[MockRobotState] = None):
        self.state = initial_state or self._create_default_state()
        self.step_count = 0
        self.simulation_time = 0.0
        self.timestep = 0.01
        self._action_history: List[np.ndarray] = []
        self._state_history: List[MockRobotState] = []
    
    def _create_default_state(self) -> MockRobotState:
        """Create default robot state."""
        return MockRobotState(
            position=np.array([0.0, 0.0, 0.8]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            joint_positions=np.zeros(35),
            joint_velocities=np.zeros(35),
            contact_forces=np.zeros(8)  # 8 contact points for feet
        )
    
    def step(self, action: Optional[np.ndarray] = None) -> MockRobotState:
        """
        Step physics simulation with optional action.
        
        Args:
            action: Joint position targets for robot control
            
        Returns:
            Updated robot state
        """
        if action is not None:
            self._action_history.append(action.copy())
            self._apply_action(action)
        
        self._update_physics()
        self._state_history.append(self._copy_state())
        
        self.step_count += 1
        self.simulation_time += self.timestep
        
        return self.state
    
    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to robot joints."""
        # Simple PD controller simulation
        target_positions = action * 0.1  # Scale actions to joint ranges
        position_error = target_positions - self.state.joint_positions
        
        # Simple proportional control
        joint_torques = position_error * 100.0  # P gain
        joint_torques -= self.state.joint_velocities * 10.0  # D gain
        
        # Update joint velocities (simplified integration)
        joint_acceleration = joint_torques / 10.0  # Simplified dynamics
        self.state.joint_velocities += joint_acceleration * self.timestep
        
        # Apply velocity limits
        max_velocity = 10.0
        self.state.joint_velocities = np.clip(
            self.state.joint_velocities, -max_velocity, max_velocity
        )
    
    def _update_physics(self) -> None:
        """Update robot physics state."""
        # Update joint positions from velocities
        self.state.joint_positions += self.state.joint_velocities * self.timestep
        
        # Simple walking simulation - forward motion
        if self.step_count > 100:  # Allow time for robot to "learn"
            forward_velocity = min(self.step_count / 1000.0, 1.0)  # Gradual speedup
            self.state.linear_velocity[0] = forward_velocity
            self.state.position += self.state.linear_velocity * self.timestep
        
        # Add some noise for realism
        noise_scale = 0.001
        self.state.position += np.random.normal(0, noise_scale, 3)
        
        # Simulate contact forces (simplified)
        if self.state.position[2] <= 0.05:  # Robot touching ground
            self.state.contact_forces = np.random.uniform(50, 200, 8)
        else:
            self.state.contact_forces = np.zeros(8)
    
    def _copy_state(self) -> MockRobotState:
        """Create a copy of current state."""
        return MockRobotState(
            position=self.state.position.copy(),
            orientation=self.state.orientation.copy(),
            linear_velocity=self.state.linear_velocity.copy(),
            angular_velocity=self.state.angular_velocity.copy(),
            joint_positions=self.state.joint_positions.copy(),
            joint_velocities=self.state.joint_velocities.copy(),
            contact_forces=self.state.contact_forces.copy() if self.state.contact_forces is not None else None,
            step_count=self.step_count,
            simulation_time=self.simulation_time
        )
    
    def reset(self, state: Optional[MockRobotState] = None) -> MockRobotState:
        """Reset physics engine to initial state."""
        self.state = state or self._create_default_state()
        self.step_count = 0
        self.simulation_time = 0.0
        self._action_history.clear()
        self._state_history.clear()
        return self.state
    
    def get_observation(self) -> np.ndarray:
        """Get observation vector from current state."""
        # Standard observation: pos(3) + quat(4) + joints(35) + joint_vel(35) + prev_action(35) + target_vel(1)
        obs_components = [
            self.state.position,
            self.state.orientation,
            self.state.joint_positions,
            self.state.joint_velocities
        ]
        
        # Previous action (zero if no history)
        if self._action_history:
            obs_components.append(self._action_history[-1])
        else:
            obs_components.append(np.zeros(35))
        
        # Target velocity
        obs_components.append(np.array([1.0]))  # Default target velocity
        
        return np.concatenate(obs_components).astype(np.float32)


@contextmanager
def mock_genesis_environment(
    initial_state: Optional[MockRobotState] = None,
    config: Optional[Dict[str, Any]] = None
) -> Generator[MockPhysicsEngine, None, None]:
    """
    Context manager for mock Genesis environment.
    
    Args:
        initial_state: Initial robot state
        config: Environment configuration
        
    Yields:
        MockPhysicsEngine instance
        
    Example:
        with mock_genesis_environment() as env:
            obs = env.get_observation()
            state = env.step(action)
    """
    config = config or {}
    engine = MockPhysicsEngine(initial_state)
    
    # Apply configuration
    if 'timestep' in config:
        engine.timestep = config['timestep']
    
    try:
        yield engine
    finally:
        # Cleanup if needed
        pass


@pytest.fixture
def mock_physics_engine():
    """Pytest fixture for mock physics engine."""
    return MockPhysicsEngine()


@pytest.fixture
def mock_robot_state():
    """Pytest fixture for mock robot state."""
    return MockRobotState(
        position=np.array([0.0, 0.0, 0.8]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        joint_positions=np.zeros(35),
        joint_velocities=np.zeros(35)
    )


@pytest.fixture
def genesis_mock_environment():
    """Pytest fixture for complete Genesis environment mock."""
    with mock_genesis_environment() as env:
        yield env


@pytest.fixture
def sample_observations(mock_physics_engine):
    """Pytest fixture for sample observation sequences."""
    observations = []
    
    # Generate realistic observation sequence
    for i in range(10):
        action = np.random.uniform(-0.1, 0.1, 35)  # Small random actions
        mock_physics_engine.step(action)
        obs = mock_physics_engine.get_observation()
        observations.append(obs)
    
    return observations


@pytest.fixture
def sample_actions():
    """Pytest fixture for sample action sequences."""
    # Generate realistic action sequence for walking
    actions = []
    
    for step in range(50):
        # Simulate walking gait pattern
        phase = (step % 20) / 20.0 * 2 * np.pi  # 20-step gait cycle
        
        # Hip oscillation
        hip_left = 0.2 * np.sin(phase)
        hip_right = 0.2 * np.sin(phase + np.pi)
        
        # Knee flexion
        knee_left = 0.3 * abs(np.sin(phase))
        knee_right = 0.3 * abs(np.sin(phase + np.pi))
        
        # Create full action vector (simplified)
        action = np.zeros(35)
        action[0] = hip_left    # Left hip
        action[1] = knee_left   # Left knee
        action[6] = hip_right   # Right hip
        action[7] = knee_right  # Right knee
        
        # Add small noise
        action += np.random.normal(0, 0.01, 35)
        
        actions.append(action)
    
    return actions


# Mock environment configurations
@pytest.fixture
def environment_configs():
    """Pytest fixture for various environment configurations."""
    return {
        'training': {
            'simulation_fps': 100,
            'control_freq': 20,
            'episode_length': 1000,
            'action_scaling': 0.1,
            'observation_noise': 0.01
        },
        'evaluation': {
            'simulation_fps': 60,
            'control_freq': 20,
            'episode_length': 2000,
            'action_scaling': 0.1,
            'observation_noise': 0.0
        },
        'curriculum_stage_1': {
            'simulation_fps': 50,
            'control_freq': 10,
            'episode_length': 500,
            'action_scaling': 0.05,
            'target_velocity': 0.0
        },
        'curriculum_stage_3': {
            'simulation_fps': 100,
            'control_freq': 20,
            'episode_length': 1500,
            'action_scaling': 0.12,
            'target_velocity': 1.5
        }
    }


# Genesis integration mocks
@pytest.fixture
def mock_genesis_scene():
    """Enhanced mock Genesis scene with realistic behavior."""
    scene = MagicMock()
    
    # Configure scene properties
    scene.sim_options = MagicMock()
    scene.viewer_options = MagicMock()
    scene.substeps = 10
    
    # Mock scene lifecycle
    scene.build.return_value = None
    scene.step.return_value = None
    scene.reset.return_value = None
    
    # Mock entity management
    mock_robot = MagicMock()
    mock_robot.n_dofs = 35
    mock_robot.n_links = 30
    mock_robot.get_pos.return_value = torch.tensor([0.0, 0.0, 0.8])
    mock_robot.get_quat.return_value = torch.tensor([0.0, 0.0, 0.0, 1.0])
    mock_robot.get_dofs_position.return_value = torch.zeros(35)
    mock_robot.get_dofs_velocity.return_value = torch.zeros(35)
    
    scene.add_entity.return_value = mock_robot
    
    return scene


@pytest.fixture
def mock_genesis_module():
    """Mock entire Genesis module for comprehensive testing."""
    with patch('genesis') as mock_gs:
        # Configure Genesis initialization
        mock_gs.init.return_value = None
        
        # Configure scene creation
        mock_scene = MagicMock()
        mock_gs.Scene.return_value = mock_scene
        
        # Configure morphs
        mock_gs.morphs.Plane.return_value = MagicMock()
        mock_gs.morphs.URDF.return_value = MagicMock()
        
        # Configure options
        mock_gs.options.SimOptions.return_value = MagicMock()
        mock_gs.options.ViewerOptions.return_value = MagicMock()
        
        yield mock_gs


class EnvironmentTestCase:
    """
    Base test case for environment-related tests.
    
    Provides common utilities and assertions for environment testing.
    """
    
    @staticmethod
    def assert_valid_observation(obs: np.ndarray, expected_size: int = 113):
        """Assert that observation vector is valid."""
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape == (expected_size,)
        assert np.all(np.isfinite(obs)), "Observation contains invalid values"
    
    @staticmethod
    def assert_valid_action(action: np.ndarray, expected_size: int = 35):
        """Assert that action vector is valid."""
        assert isinstance(action, np.ndarray)
        assert action.shape == (expected_size,)
        # Actions should be reasonable for robot control
        assert np.all(np.abs(action) <= 2.0), "Action values too extreme"
    
    @staticmethod
    def assert_valid_reward(reward: float):
        """Assert that reward value is valid."""
        assert isinstance(reward, (int, float))
        assert np.isfinite(reward), "Reward is not finite"
    
    @staticmethod
    def assert_physics_stability(state: MockRobotState):
        """Assert that physics state is stable."""
        # Check for reasonable position bounds
        assert np.all(np.abs(state.position[:2]) < 50.0), "Robot position too extreme"
        assert 0.0 <= state.position[2] <= 5.0, "Robot height unrealistic"
        
        # Check for reasonable velocities
        assert np.all(np.abs(state.linear_velocity) < 20.0), "Linear velocity too extreme"
        assert np.all(np.abs(state.angular_velocity) < 50.0), "Angular velocity too extreme"
        assert np.all(np.abs(state.joint_velocities) < 100.0), "Joint velocities too extreme"
        
        # Check for no NaN values
        assert np.all(np.isfinite(state.position)), "Position contains NaN/inf"
        assert np.all(np.isfinite(state.joint_positions)), "Joint positions contain NaN/inf"


@pytest.fixture
def environment_test_case():
    """Pytest fixture for environment test case utilities."""
    return EnvironmentTestCase()


# Performance testing fixtures
@pytest.fixture
def performance_simulation_config():
    """Configuration for performance testing simulations."""
    return {
        'num_episodes': 100,
        'max_steps_per_episode': 1000,
        'target_fps': 100,
        'batch_size': 8,
        'parallel_environments': 4
    }