"""
Anti-corruption layer for Genesis physics engine.
Translates between domain concepts and Genesis-specific implementation.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

from ...domain.model.value_objects import (
    MotionCommand,
    MotionType,
    GaitPattern,
    MovementTrajectory,
)
from ...domain.model.aggregates import HumanoidRobot
from ...protocols import RobotState
from ...physics.physics_manager import GenesisPhysicsManager

logger = logging.getLogger(__name__)


class GenesisSimulationAdapter:
    """
    Anti-corruption layer for Genesis physics engine.

    Protects domain model from Genesis-specific details and provides
    domain-focused interface for physics simulation operations.
    """

    def __init__(self, genesis_manager: GenesisPhysicsManager):
        self._genesis = genesis_manager
        self._motion_command_cache: Dict[str, np.ndarray] = {}
        self._trajectory_history: List[MovementTrajectory] = []

    def execute_motion_command(
        self, robot: HumanoidRobot, command: MotionCommand
    ) -> Dict[str, Any]:
        """
        Execute domain motion command through Genesis.

        Translates high-level motion commands to low-level control actions.
        """
        try:
            # Translate motion command to Genesis actions
            genesis_actions = self._translate_motion_command(command, robot)

            # Apply control through Genesis
            self._genesis.apply_robot_control(genesis_actions)

            # Get updated robot state
            genesis_state = self._genesis.get_robot_state()
            domain_state = self._translate_robot_state(genesis_state)

            return {
                "success": True,
                "robot_state": domain_state,
                "command_executed": command,
                "genesis_actions": genesis_actions.tolist(),
            }

        except Exception as e:
            logger.error(f"Failed to execute motion command {command.motion_type}: {e}")
            return {"success": False, "error": str(e), "command_attempted": command}

    def simulate_episode_step(self, steps: int = 1) -> Dict[str, Any]:
        """
        Advance physics simulation with domain-focused interface.
        """
        try:
            # Execute Genesis simulation steps
            self._genesis.step_simulation(steps)

            # Get current world state
            robot_state = self._genesis.get_robot_state()
            domain_state = self._translate_robot_state(robot_state)

            # Assess physics stability
            stability_assessment = self._assess_physics_stability(domain_state)

            return {
                "success": True,
                "steps_executed": steps,
                "robot_state": domain_state,
                "physics_stable": stability_assessment["stable"],
                "stability_score": stability_assessment["score"],
            }

        except Exception as e:
            logger.error(f"Failed to simulate episode step: {e}")
            return {"success": False, "error": str(e), "steps_attempted": steps}

    def extract_gait_pattern(self, trajectory: MovementTrajectory) -> GaitPattern:
        """
        Extract gait pattern from movement trajectory using domain logic.

        Hides Genesis-specific gait analysis behind domain interface.
        """
        # Convert trajectory to Genesis-compatible format
        genesis_trajectory = self._convert_trajectory_to_genesis(trajectory)

        # Perform gait analysis (simplified - could use more sophisticated Genesis features)
        stride_analysis = self._analyze_stride_pattern(trajectory)
        stability_analysis = self._analyze_gait_stability(trajectory)

        return GaitPattern(
            stride_length=stride_analysis["stride_length"],
            stride_frequency=stride_analysis["stride_frequency"],
            step_height=stride_analysis["step_height"],
            stability_margin=stability_analysis["stability_margin"],
            energy_efficiency=stability_analysis["energy_efficiency"],
            symmetry_score=stability_analysis["symmetry_score"],
        )

    def assess_robot_capabilities(self, robot: HumanoidRobot) -> Dict[str, Any]:
        """
        Assess robot capabilities through Genesis simulation.

        Provides domain-focused capability assessment.
        """
        capabilities = {
            "joint_count": robot.joint_count,
            "simulated_mass": robot.weight,
            "simulated_height": robot.height,
            "balance_capability": 0.0,
            "locomotion_capability": 0.0,
            "stability_rating": "unknown",
        }

        try:
            # Test basic balance capability
            balance_score = self._test_balance_capability()
            capabilities["balance_capability"] = balance_score

            # Test locomotion capability
            locomotion_score = self._test_locomotion_capability()
            capabilities["locomotion_capability"] = locomotion_score

            # Overall stability rating
            overall_score = (balance_score + locomotion_score) / 2.0
            if overall_score >= 0.8:
                capabilities["stability_rating"] = "excellent"
            elif overall_score >= 0.6:
                capabilities["stability_rating"] = "good"
            elif overall_score >= 0.4:
                capabilities["stability_rating"] = "fair"
            else:
                capabilities["stability_rating"] = "poor"

        except Exception as e:
            logger.warning(f"Failed to assess robot capabilities: {e}")
            capabilities["assessment_error"] = str(e)

        return capabilities

    def get_simulation_diagnostics(self) -> Dict[str, Any]:
        """
        Get simulation diagnostics in domain terms.
        """
        try:
            genesis_info = self._genesis.get_simulation_info()

            return {
                "simulation_stable": genesis_info.get("scene_initialized", False),
                "robot_loaded": genesis_info.get("robot_loaded", False),
                "control_responsive": True,  # Could add actual responsiveness test
                "physics_engine": "Genesis",
                "action_scale": genesis_info.get("action_scale", 0.1),
                "control_frequency": genesis_info.get("control_frequency", 20),
                "diagnostics_timestamp": str(np.datetime64("now")),
            }

        except Exception as e:
            return {
                "simulation_stable": False,
                "error": str(e),
                "diagnostics_timestamp": str(np.datetime64("now")),
            }

    # Private translation methods

    def _translate_motion_command(
        self, command: MotionCommand, robot: HumanoidRobot
    ) -> np.ndarray:
        """Translate domain motion command to Genesis action vector."""
        # Cap velocity early to prevent double scaling
        capped_velocity = min(command.velocity, 2.0)

        # Cache key for motion command using capped velocity
        cache_key = f"{command.motion_type.value}_{capped_velocity}_{robot.joint_count}"

        if cache_key in self._motion_command_cache:
            return self._motion_command_cache[cache_key]

        # Generate action based on motion type
        action = np.zeros(robot.joint_count, dtype=np.float32)

        if command.motion_type == MotionType.BALANCE:
            # Minimal joint movements for balance
            action = np.random.normal(0, 0.02, robot.joint_count)

        elif command.motion_type == MotionType.WALK_FORWARD:
            # Forward walking pattern (simplified)
            # In practice, this would be more sophisticated
            action = self._generate_walking_pattern(capped_velocity, "forward")

        elif command.motion_type == MotionType.WALK_BACKWARD:
            action = self._generate_walking_pattern(capped_velocity, "backward")

        elif command.motion_type == MotionType.TURN_LEFT:
            action = self._generate_turning_pattern(capped_velocity, "left")

        elif command.motion_type == MotionType.TURN_RIGHT:
            action = self._generate_turning_pattern(capped_velocity, "right")

        elif command.motion_type == MotionType.STOP:
            # All joints to neutral position
            action = np.zeros(robot.joint_count)

        # Velocity scaling is already applied in pattern generation methods
        # No additional scaling needed since we use capped_velocity

        # Clip to safe range
        action = np.clip(action, -1.0, 1.0)

        # Cache the result
        self._motion_command_cache[cache_key] = action

        return action

    def _translate_robot_state(self, genesis_state: RobotState) -> RobotState:
        """Translate Genesis robot state to domain robot state."""
        # In this case, the RobotState is already in the right format
        # But we could add domain-specific validation or transformation here
        return genesis_state

    def _assess_physics_stability(self, robot_state: RobotState) -> Dict[str, Any]:
        """Assess physics simulation stability."""
        stability_score = 1.0
        issues = []

        # Check for unrealistic values
        if not np.all(np.isfinite(robot_state.position)):
            stability_score -= 0.5
            issues.append("Invalid position values")

        if not np.all(np.isfinite(robot_state.joint_positions)):
            stability_score -= 0.3
            issues.append("Invalid joint positions")

        # Check for extreme values
        if robot_state.position[2] < 0 or robot_state.position[2] > 3.0:
            stability_score -= 0.2
            issues.append("Extreme height values")

        if np.any(np.abs(robot_state.joint_velocities) > 50.0):
            stability_score -= 0.2
            issues.append("Extreme joint velocities")

        return {
            "stable": stability_score >= 0.7,
            "score": max(0.0, stability_score),
            "issues": issues,
        }

    def _convert_trajectory_to_genesis(
        self, trajectory: MovementTrajectory
    ) -> Dict[str, Any]:
        """Convert domain trajectory to Genesis-compatible format."""
        return {
            "positions": trajectory.positions,
            "timestamps": trajectory.timestamps,
            "velocities": trajectory.velocities,
        }

    def _analyze_stride_pattern(
        self, trajectory: MovementTrajectory
    ) -> Dict[str, float]:
        """Analyze stride pattern from trajectory."""
        if len(trajectory.positions) < 3:
            # Return minimum valid values to satisfy GaitPattern validation
            return {
                "stride_length": 0.01,  # Minimum valid stride length
                "stride_frequency": 0.1,  # Minimum valid frequency
                "step_height": 0.01,  # Minimum valid step height
            }

        # Simple stride analysis
        total_distance = trajectory.get_total_distance()
        total_time = trajectory.timestamps[-1] - trajectory.timestamps[0]

        if total_time <= 0:
            # Return minimum valid values for invalid time duration
            return {
                "stride_length": 0.01,  # Minimum valid stride length
                "stride_frequency": 0.1,  # Minimum valid frequency
                "step_height": 0.01,  # Minimum valid step height
            }

        avg_velocity = total_distance / total_time
        estimated_stride_freq = max(avg_velocity / 0.5, 0.1)  # Assume 0.5m stride
        # Ensure minimum stride length to satisfy GaitPattern validation
        estimated_stride_length = max(
            avg_velocity / estimated_stride_freq if estimated_stride_freq > 0 else 0.01,
            0.01,  # Minimum valid stride length
        )

        return {
            "stride_length": estimated_stride_length,
            "stride_frequency": estimated_stride_freq,
            "step_height": max(0.05, 0.01),  # Ensure positive step height
        }

    def _analyze_gait_stability(
        self, trajectory: MovementTrajectory
    ) -> Dict[str, float]:
        """Analyze gait stability from trajectory."""
        smoothness = trajectory.get_smoothness_score()

        return {
            "stability_margin": smoothness * 0.1,  # Convert to meters
            "energy_efficiency": smoothness,
            "symmetry_score": min(smoothness + 0.2, 1.0),
        }

    def _generate_walking_pattern(self, velocity: float, direction: str) -> np.ndarray:
        """Generate walking pattern for specified direction."""
        # Simplified walking pattern generation
        # In practice, this would use learned patterns or biomechanical models
        pattern = np.random.normal(0, 0.1, 35)  # Assuming 35 joints

        if direction == "forward":
            # Hip and knee patterns for forward walking
            pattern[6:12] = np.array([0.1, -0.2, 0.1, 0.1, -0.2, 0.1]) * velocity
        elif direction == "backward":
            pattern[6:12] = np.array([-0.1, 0.2, -0.1, -0.1, 0.2, -0.1]) * velocity

        return pattern

    def _generate_turning_pattern(self, velocity: float, direction: str) -> np.ndarray:
        """Generate turning pattern for specified direction."""
        pattern = np.random.normal(0, 0.05, 35)

        if direction == "left":
            # Asymmetric leg patterns for left turn
            pattern[6:9] = np.array([0.1, -0.1, 0.05]) * velocity  # Left leg
            pattern[9:12] = np.array([0.15, -0.15, 0.1]) * velocity  # Right leg
        elif direction == "right":
            pattern[6:9] = np.array([0.15, -0.15, 0.1]) * velocity  # Left leg
            pattern[9:12] = np.array([0.1, -0.1, 0.05]) * velocity  # Right leg

        return pattern

    def _test_balance_capability(self) -> float:
        """Test robot's balance capability."""
        # Simplified balance test
        # Could implement actual balance perturbation tests
        return 0.8  # Placeholder

    def _test_locomotion_capability(self) -> float:
        """Test robot's locomotion capability."""
        # Simplified locomotion test
        # Could implement actual walking tests
        return 0.7  # Placeholder
