"""
Comprehensive tests for Genesis simulation adapter.
Tests anti-corruption layer functionality and Genesis integration.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.genesis_humanoid_rl.infrastructure.adapters.genesis_adapter import GenesisSimulationAdapter
from src.genesis_humanoid_rl.domain.model.value_objects import (
    MotionCommand, MotionType, GaitPattern, MovementTrajectory, PerformanceMetrics
)
from src.genesis_humanoid_rl.domain.model.aggregates import HumanoidRobot, RobotType
from src.genesis_humanoid_rl.protocols import RobotState


class TestGenesisSimulationAdapter:
    """Test GenesisSimulationAdapter anti-corruption layer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock Genesis physics manager
        self.mock_genesis = Mock()
        self.mock_genesis.get_simulation_info.return_value = {
            'scene_initialized': True,
            'robot_loaded': True,
            'action_scale': 0.1,
            'control_frequency': 20
        }
        
        # Create adapter under test
        self.adapter = GenesisSimulationAdapter(self.mock_genesis)
        
        # Create test robot
        self.robot = HumanoidRobot(
            robot_id=Mock(),
            robot_type=RobotType.UNITREE_G1,
            name="Test Robot",
            joint_count=35,
            height=1.2,
            weight=35.0
        )
        
        # Create test robot state
        self.test_robot_state = RobotState(
            position=np.array([0.0, 0.0, 0.8]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            joint_positions=np.zeros(35),
            joint_velocities=np.zeros(35),
            timestamp=0.0
        )
        
        self.mock_genesis.get_robot_state.return_value = self.test_robot_state
    
    def test_execute_motion_command_successful(self):
        """Test successful motion command execution."""
        command = MotionCommand(
            motion_type=MotionType.WALK_FORWARD,
            velocity=1.0,
            duration=2.0
        )
        
        # Mock successful Genesis execution
        self.mock_genesis.apply_robot_control.return_value = None
        
        result = self.adapter.execute_motion_command(self.robot, command)
        
        assert result['success'] is True
        assert result['robot_state'] == self.test_robot_state
        assert result['command_executed'] == command
        assert 'genesis_actions' in result
        assert isinstance(result['genesis_actions'], list)
        
        # Verify Genesis was called with translated action
        self.mock_genesis.apply_robot_control.assert_called_once()
        call_args = self.mock_genesis.apply_robot_control.call_args[0][0]
        assert isinstance(call_args, np.ndarray)
        assert len(call_args) == self.robot.joint_count
    
    def test_execute_motion_command_genesis_failure(self):
        """Test motion command execution when Genesis fails."""
        command = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        
        # Mock Genesis failure
        self.mock_genesis.apply_robot_control.side_effect = RuntimeError("Genesis simulation error")
        
        result = self.adapter.execute_motion_command(self.robot, command)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['command_attempted'] == command
        assert "Genesis simulation error" in result['error']
    
    def test_motion_command_translation_balance(self):
        """Test translation of balance motion command."""
        balance_cmd = MotionCommand(MotionType.BALANCE, velocity=0.5)
        
        # Access private method for testing
        action = self.adapter._translate_motion_command(balance_cmd, self.robot)
        
        assert isinstance(action, np.ndarray)
        assert len(action) == self.robot.joint_count
        assert np.all(action >= -1.0) and np.all(action <= 1.0)  # Clipped to safe range
        
        # Should be small movements for balance
        assert np.all(np.abs(action) < 0.5)
    
    def test_motion_command_translation_walking(self):
        """Test translation of walking motion commands."""
        forward_cmd = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        backward_cmd = MotionCommand(MotionType.WALK_BACKWARD, velocity=1.0)
        
        forward_action = self.adapter._translate_motion_command(forward_cmd, self.robot)
        backward_action = self.adapter._translate_motion_command(backward_cmd, self.robot)
        
        # Actions should be different for forward vs backward
        assert not np.allclose(forward_action, backward_action)
        
        # Both should be properly scaled and clipped
        for action in [forward_action, backward_action]:
            assert isinstance(action, np.ndarray)
            assert len(action) == self.robot.joint_count
            assert np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def test_motion_command_translation_turning(self):
        """Test translation of turning motion commands."""
        left_cmd = MotionCommand(MotionType.TURN_LEFT, velocity=1.0)
        right_cmd = MotionCommand(MotionType.TURN_RIGHT, velocity=1.0)
        
        left_action = self.adapter._translate_motion_command(left_cmd, self.robot)
        right_action = self.adapter._translate_motion_command(right_cmd, self.robot)
        
        # Actions should be different for left vs right turns
        assert not np.allclose(left_action, right_action)
        
        # Should have asymmetric patterns (different for left/right legs)
        assert not np.allclose(left_action[6:9], left_action[9:12])  # Asymmetric legs
    
    def test_motion_command_translation_stop(self):
        """Test translation of stop motion command."""
        stop_cmd = MotionCommand(MotionType.STOP, velocity=0.0)
        
        action = self.adapter._translate_motion_command(stop_cmd, self.robot)
        
        # Stop command should result in zero actions
        assert np.allclose(action, 0.0)
    
    def test_motion_command_velocity_scaling(self):
        """Test velocity scaling in motion command translation."""
        cmd_slow = MotionCommand(MotionType.WALK_FORWARD, velocity=0.5)
        cmd_fast = MotionCommand(MotionType.WALK_FORWARD, velocity=2.0)
        cmd_extreme = MotionCommand(MotionType.WALK_FORWARD, velocity=5.0)
        
        action_slow = self.adapter._translate_motion_command(cmd_slow, self.robot)
        action_fast = self.adapter._translate_motion_command(cmd_fast, self.robot)
        action_extreme = self.adapter._translate_motion_command(cmd_extreme, self.robot)
        
        # Higher velocity should generally result in larger actions (up to cap)
        slow_magnitude = np.linalg.norm(action_slow)
        fast_magnitude = np.linalg.norm(action_fast)
        extreme_magnitude = np.linalg.norm(action_extreme)
        
        assert slow_magnitude < fast_magnitude
        # Extreme velocity should be capped at 2x, so similar to fast at 2.0
        assert np.isclose(fast_magnitude, extreme_magnitude, rtol=0.1)
    
    def test_motion_command_caching(self):
        """Test motion command translation caching."""
        cmd = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        
        # First translation
        action1 = self.adapter._translate_motion_command(cmd, self.robot)
        
        # Second translation (should use cache)
        action2 = self.adapter._translate_motion_command(cmd, self.robot)
        
        # Should be identical due to caching
        assert np.array_equal(action1, action2)
        
        # Check cache contains the result
        cache_key = f"{cmd.motion_type.value}_{cmd.velocity}_{self.robot.joint_count}"
        assert cache_key in self.adapter._motion_command_cache
    
    def test_simulate_episode_step_successful(self):
        """Test successful episode step simulation."""
        steps = 5
        
        result = self.adapter.simulate_episode_step(steps)
        
        assert result['success'] is True
        assert result['steps_executed'] == steps
        assert result['robot_state'] == self.test_robot_state
        assert 'physics_stable' in result
        assert 'stability_score' in result
        
        # Verify Genesis was called
        self.mock_genesis.step_simulation.assert_called_once_with(steps)
        self.mock_genesis.get_robot_state.assert_called()
    
    def test_simulate_episode_step_genesis_failure(self):
        """Test episode step simulation when Genesis fails."""
        # Mock Genesis failure
        self.mock_genesis.step_simulation.side_effect = RuntimeError("Physics instability")
        
        result = self.adapter.simulate_episode_step(3)
        
        assert result['success'] is False
        assert 'error' in result
        assert result['steps_attempted'] == 3
        assert "Physics instability" in result['error']
    
    def test_physics_stability_assessment_stable(self):
        """Test physics stability assessment for stable robot state."""
        stable_state = RobotState(
            position=np.array([0.0, 0.0, 0.8]),  # Normal height
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # Upright
            joint_positions=np.zeros(35),  # Normal joint positions
            joint_velocities=np.ones(35) * 2.0,  # Reasonable velocities
            timestamp=0.0
        )
        
        assessment = self.adapter._assess_physics_stability(stable_state)
        
        assert assessment['stable'] is True
        assert assessment['score'] >= 0.7
        assert len(assessment['issues']) == 0
    
    def test_physics_stability_assessment_unstable(self):
        """Test physics stability assessment for unstable robot state."""
        unstable_state = RobotState(
            position=np.array([0.0, 0.0, -0.5]),  # Underground
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            joint_positions=np.full(35, np.inf),  # Invalid positions
            joint_velocities=np.ones(35) * 100.0,  # Extreme velocities
            timestamp=0.0
        )
        
        assessment = self.adapter._assess_physics_stability(unstable_state)
        
        assert assessment['stable'] is False
        assert assessment['score'] < 0.7
        assert len(assessment['issues']) > 0
        assert any("Invalid" in issue for issue in assessment['issues'])
        assert any("Extreme" in issue for issue in assessment['issues'])
    
    def test_extract_gait_pattern_from_trajectory(self):
        """Test gait pattern extraction from movement trajectory."""
        trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0, 2.0],
            velocities=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        )
        
        gait_pattern = self.adapter.extract_gait_pattern(trajectory)
        
        assert isinstance(gait_pattern, GaitPattern)
        assert gait_pattern.stride_length > 0
        assert gait_pattern.stride_frequency > 0
        assert gait_pattern.step_height >= 0
        assert 0.0 <= gait_pattern.energy_efficiency <= 1.0
        assert 0.0 <= gait_pattern.symmetry_score <= 1.0
        assert 0.0 <= gait_pattern.stability_margin <= 1.0
    
    def test_extract_gait_pattern_insufficient_data(self):
        """Test gait pattern extraction with insufficient trajectory data."""
        short_trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0]
        )
        
        gait_pattern = self.adapter.extract_gait_pattern(short_trajectory)
        
        # Should still return valid gait pattern with defaults
        assert isinstance(gait_pattern, GaitPattern)
        assert gait_pattern.stride_length >= 0
        assert gait_pattern.stride_frequency >= 0
    
    def test_assess_robot_capabilities(self):
        """Test robot capabilities assessment."""
        capabilities = self.adapter.assess_robot_capabilities(self.robot)
        
        assert isinstance(capabilities, dict)
        assert capabilities['joint_count'] == self.robot.joint_count
        assert capabilities['simulated_mass'] == self.robot.weight
        assert capabilities['simulated_height'] == self.robot.height
        assert 'balance_capability' in capabilities
        assert 'locomotion_capability' in capabilities
        assert 'stability_rating' in capabilities
        
        # Check value ranges
        assert 0.0 <= capabilities['balance_capability'] <= 1.0
        assert 0.0 <= capabilities['locomotion_capability'] <= 1.0
        assert capabilities['stability_rating'] in ['excellent', 'good', 'fair', 'poor']
    
    def test_assess_robot_capabilities_with_error(self):
        """Test robot capabilities assessment when tests fail."""
        # Make capability tests fail
        with patch.object(self.adapter, '_test_balance_capability', side_effect=RuntimeError("Test failed")):
            capabilities = self.adapter.assess_robot_capabilities(self.robot)
        
        assert 'assessment_error' in capabilities
        assert "Test failed" in capabilities['assessment_error']
    
    def test_get_simulation_diagnostics_successful(self):
        """Test simulation diagnostics retrieval."""
        diagnostics = self.adapter.get_simulation_diagnostics()
        
        assert isinstance(diagnostics, dict)
        assert diagnostics['simulation_stable'] is True
        assert diagnostics['robot_loaded'] is True
        assert diagnostics['control_responsive'] is True
        assert diagnostics['physics_engine'] == 'Genesis'
        assert 'action_scale' in diagnostics
        assert 'control_frequency' in diagnostics
        assert 'diagnostics_timestamp' in diagnostics
    
    def test_get_simulation_diagnostics_with_error(self):
        """Test simulation diagnostics when Genesis info fails."""
        # Mock Genesis info failure
        self.mock_genesis.get_simulation_info.side_effect = RuntimeError("Info unavailable")
        
        diagnostics = self.adapter.get_simulation_diagnostics()
        
        assert diagnostics['simulation_stable'] is False
        assert 'error' in diagnostics
        assert "Info unavailable" in diagnostics['error']
        assert 'diagnostics_timestamp' in diagnostics
    
    def test_stride_analysis_valid_trajectory(self):
        """Test stride pattern analysis with valid trajectory."""
        trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8), (3.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0, 2.0, 3.0]
        )
        
        stride_analysis = self.adapter._analyze_stride_pattern(trajectory)
        
        assert isinstance(stride_analysis, dict)
        assert 'stride_length' in stride_analysis
        assert 'stride_frequency' in stride_analysis
        assert 'step_height' in stride_analysis
        
        # Should calculate reasonable values
        assert stride_analysis['stride_length'] > 0
        assert stride_analysis['stride_frequency'] > 0
        assert stride_analysis['step_height'] >= 0
    
    def test_stride_analysis_zero_time(self):
        """Test stride pattern analysis with zero time duration."""
        trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)],
            timestamps=[1.0, 1.0]  # Same timestamp
        )
        
        stride_analysis = self.adapter._analyze_stride_pattern(trajectory)
        
        # Should handle zero time gracefully with minimum valid values
        assert stride_analysis['stride_length'] == 0.01  # Minimum valid value
        assert stride_analysis['stride_frequency'] == 0.1  # Minimum valid value
        assert stride_analysis['step_height'] > 0  # Positive value
    
    def test_gait_stability_analysis(self):
        """Test gait stability analysis."""
        smooth_trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0, 2.0]
        )
        
        stability_analysis = self.adapter._analyze_gait_stability(smooth_trajectory)
        
        assert isinstance(stability_analysis, dict)
        assert 'stability_margin' in stability_analysis
        assert 'energy_efficiency' in stability_analysis
        assert 'symmetry_score' in stability_analysis
        
        # All values should be in valid ranges
        assert 0.0 <= stability_analysis['stability_margin'] <= 1.0
        assert 0.0 <= stability_analysis['energy_efficiency'] <= 1.0
        assert 0.0 <= stability_analysis['symmetry_score'] <= 1.0
    
    def test_walking_pattern_generation_directions(self):
        """Test walking pattern generation for different directions."""
        forward_pattern = self.adapter._generate_walking_pattern(1.0, 'forward')
        backward_pattern = self.adapter._generate_walking_pattern(1.0, 'backward')
        
        assert len(forward_pattern) == 35  # Expected joint count
        assert len(backward_pattern) == 35
        
        # Patterns should be different for different directions
        assert not np.allclose(forward_pattern, backward_pattern)
        
        # Hip and knee joints should have specific patterns
        assert not np.allclose(forward_pattern[6:12], backward_pattern[6:12])
    
    def test_turning_pattern_generation_directions(self):
        """Test turning pattern generation for different directions."""
        left_pattern = self.adapter._generate_turning_pattern(1.0, 'left')
        right_pattern = self.adapter._generate_turning_pattern(1.0, 'right')
        
        assert len(left_pattern) == 35
        assert len(right_pattern) == 35
        
        # Patterns should be different for different directions
        assert not np.allclose(left_pattern, right_pattern)
        
        # Should have asymmetric leg patterns
        assert not np.allclose(left_pattern[6:9], left_pattern[9:12])
        assert not np.allclose(right_pattern[6:9], right_pattern[9:12])
    
    def test_capability_test_placeholders(self):
        """Test capability test placeholder methods."""
        balance_score = self.adapter._test_balance_capability()
        locomotion_score = self.adapter._test_locomotion_capability()
        
        # Placeholder methods should return valid scores
        assert 0.0 <= balance_score <= 1.0
        assert 0.0 <= locomotion_score <= 1.0
    
    def test_robot_state_translation_passthrough(self):
        """Test robot state translation (currently passthrough)."""
        genesis_state = self.test_robot_state
        domain_state = self.adapter._translate_robot_state(genesis_state)
        
        # Currently a passthrough, should be identical
        assert domain_state == genesis_state
    
    def test_trajectory_conversion_to_genesis(self):
        """Test trajectory conversion to Genesis format."""
        trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0],
            velocities=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        )
        
        genesis_format = self.adapter._convert_trajectory_to_genesis(trajectory)
        
        assert isinstance(genesis_format, dict)
        assert 'positions' in genesis_format
        assert 'timestamps' in genesis_format
        assert 'velocities' in genesis_format
        
        assert genesis_format['positions'] == trajectory.positions
        assert genesis_format['timestamps'] == trajectory.timestamps
        assert genesis_format['velocities'] == trajectory.velocities