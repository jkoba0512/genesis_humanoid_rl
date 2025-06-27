"""
Test configuration and fixtures for genesis_humanoid_rl.

Provides comprehensive test infrastructure with fixtures, context managers,
and utilities for all layers of the application.
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from typing import Any, Dict, Optional
from dataclasses import dataclass

# Import all fixture modules
from tests.fixtures.database_fixtures import *
from tests.fixtures.domain_fixtures import *
from tests.fixtures.simulation_fixtures import *


# Backward compatibility - maintain existing fixtures
@dataclass
class MockRobotState:
    """Mock robot state for testing (legacy compatibility)."""
    position: np.ndarray
    orientation: np.ndarray  # quaternion
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    n_dofs: int = 35
    n_links: int = 30


@pytest.fixture
def mock_robot():
    """
    Mock robot entity for testing without Genesis dependency.
    
    Returns a properly configured mock that behaves like a Genesis robot.
    """
    robot = MagicMock()
    
    # Configure robot properties
    robot.n_dofs = 35
    robot.n_links = 30
    
    # Mock position and orientation methods
    robot.get_pos.return_value = torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32)
    robot.get_quat.return_value = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    
    # Mock joint state methods
    robot.get_dofs_position.return_value = torch.zeros(35, dtype=torch.float32)
    robot.get_dofs_velocity.return_value = torch.zeros(35, dtype=torch.float32)
    
    # Mock control methods
    robot.control_dofs_position.return_value = None
    robot.set_pos.return_value = None
    
    return robot


@pytest.fixture
def mock_scene():
    """Mock Genesis scene for testing."""
    scene = MagicMock()
    
    # Configure scene methods
    scene.step.return_value = None
    scene.build.return_value = None
    scene.add_entity.return_value = MagicMock()  # Return mock robot
    scene.close.return_value = None
    
    return scene


@pytest.fixture
def mock_genesis():
    """Mock Genesis module for testing."""
    with patch('genesis as gs') as mock_gs:
        # Configure Genesis initialization
        mock_gs.init.return_value = None
        
        # Configure scene creation
        mock_gs.Scene.return_value = MagicMock()
        
        # Configure morphs
        mock_gs.morphs.Plane.return_value = MagicMock()
        mock_gs.morphs.URDF.return_value = MagicMock()
        
        # Configure options
        mock_gs.options.SimOptions.return_value = MagicMock()
        mock_gs.options.ViewerOptions.return_value = MagicMock()
        
        yield mock_gs


@pytest.fixture
def sample_observation():
    """Sample observation vector for testing."""
    # Observation structure: pos(3) + quat(4) + joints(35) + joint_vel(35) + prev_action(35) + target_vel(1)
    obs_size = 3 + 4 + 35 + 35 + 35 + 1  # 113 dimensions
    return np.random.randn(obs_size).astype(np.float32)


@pytest.fixture
def sample_action():
    """Sample action vector for testing."""
    return np.random.uniform(-1.0, 1.0, size=35).astype(np.float32)


@pytest.fixture
def robot_state_fixture():
    """Sample robot state for testing."""
    return MockRobotState(
        position=np.array([0.0, 0.0, 0.8]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # quaternion (x, y, z, w)
        joint_positions=np.zeros(35),
        joint_velocities=np.zeros(35),
    )


@pytest.fixture
def environment_config():
    """Sample environment configuration for testing."""
    return {
        'simulation_fps': 100,
        'control_freq': 20,
        'episode_length': 1000,
        'target_velocity': 1.0,
        'num_joints': 35,
    }


@pytest.fixture
def reward_components():
    """Sample reward components for testing."""
    return {
        'velocity_reward': 0.8,
        'stability_reward': 0.6,
        'height_reward': 0.7,
        'energy_penalty': -0.1,
        'smoothness_penalty': -0.05,
        'total_reward': 1.95,
    }


class MockPhysicsManager:
    """Mock physics manager for testing environment orchestration."""
    
    def __init__(self):
        self.scene = None
        self.robot = None
        self._step_count = 0
    
    def initialize_scene(self, config: Dict[str, Any]) -> None:
        """Mock scene initialization."""
        self.scene = MagicMock()
        
    def add_robot(self, urdf_path: str, position: np.ndarray) -> Any:
        """Mock robot addition."""
        self.robot = MagicMock()
        self.robot.n_dofs = 35
        self.robot.get_pos.return_value = torch.tensor(position, dtype=torch.float32)
        return self.robot
    
    def step_simulation(self, steps: int = 1) -> None:
        """Mock simulation stepping."""
        self._step_count += steps
    
    def get_robot_state(self) -> MockRobotState:
        """Mock robot state retrieval."""
        return MockRobotState(
            position=np.array([0.0, 0.0, 0.8]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            joint_positions=np.zeros(35),
            joint_velocities=np.zeros(35),
        )
    
    def reset_scene(self) -> None:
        """Mock scene reset."""
        self._step_count = 0


@pytest.fixture
def mock_physics_manager():
    """Mock physics manager fixture."""
    return MockPhysicsManager()


# Test utilities - maintain backward compatibility
def assert_valid_observation(obs: np.ndarray, expected_size: int = 113):
    """Assert that observation has valid structure."""
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (expected_size,)
    assert np.all(np.isfinite(obs)), "Observation contains invalid values"


def assert_valid_action(action: np.ndarray, expected_size: int = 35):
    """Assert that action has valid structure."""
    assert isinstance(action, np.ndarray)
    assert action.shape == (expected_size,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0), "Action values outside [-1, 1] range"


def assert_valid_reward(reward: float):
    """Assert that reward is valid."""
    assert isinstance(reward, (int, float))
    assert np.isfinite(reward), "Reward is not finite"


# Performance testing utilities
@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {
        'num_episodes': 10,
        'max_steps_per_episode': 100,
        'target_fps': 50,
        'memory_limit_mb': 1000,
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (slower, with dependencies)"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "database: marks tests that require database"
    )
    config.addinivalue_line(
        "markers", "simulation: marks tests that require physics simulation"
    )


# Session-level fixtures for expensive setup
@pytest.fixture(scope="session")
def shared_test_database():
    """Session-scoped database for tests that can share data."""
    with temporary_database("shared_test.db") as db_path:
        with database_connection(db_path) as db:
            # Initialize with common test data
            yield db


# Automatic cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_environment():
    """Automatically clean up environment after each test."""
    yield
    # Cleanup code here - clear any global state, close connections, etc.
    pass