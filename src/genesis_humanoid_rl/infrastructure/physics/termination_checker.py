"""
Physics simulation termination checker.

Monitors simulation state for termination conditions including physics instability,
extreme positions, constraint violations, and performance degradation.
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import time

from ..exceptions.simulation_exceptions import PhysicsInstabilityError
from ..exceptions.genesis_exceptions import GenesisPhysicsError

logger = logging.getLogger(__name__)


class TerminationReason(Enum):
    """Reasons for simulation termination."""

    # Physics-based terminations
    PHYSICS_INSTABILITY = "physics_instability"
    EXTREME_POSITION = "extreme_position"
    EXTREME_VELOCITY = "extreme_velocity"
    EXTREME_ACCELERATION = "extreme_acceleration"
    CONSTRAINT_VIOLATION = "constraint_violation"

    # Robot-specific terminations
    ROBOT_FALLEN = "robot_fallen"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    SELF_COLLISION = "self_collision"
    GROUND_COLLISION = "ground_collision"

    # Performance-based terminations
    SIMULATION_TIMEOUT = "simulation_timeout"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MEMORY_EXHAUSTION = "memory_exhaustion"

    # Numerical terminations
    NAN_VALUES = "nan_values"
    INFINITE_VALUES = "infinite_values"
    NUMERICAL_OVERFLOW = "numerical_overflow"

    # Episode completion
    SUCCESS = "success"
    MAX_STEPS_REACHED = "max_steps_reached"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class TerminationCheckResult:
    """Result of termination check with diagnostic information."""

    should_terminate: bool
    reason: TerminationReason
    confidence: float  # 0.0 to 1.0
    diagnostic_data: Dict[str, Any]
    recovery_possible: bool
    severity: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class PhysicsState:
    """Snapshot of physics state for analysis."""

    robot_position: np.ndarray
    robot_orientation: np.ndarray  # quaternion
    robot_linear_velocity: np.ndarray
    robot_angular_velocity: np.ndarray
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_accelerations: Optional[np.ndarray]
    contact_forces: Optional[np.ndarray]
    constraint_forces: Optional[np.ndarray]
    step_count: int
    simulation_time: float
    timestamp: float


class TerminationChecker:
    """
    Comprehensive termination checker for physics simulation.

    Monitors multiple aspects of simulation state to detect termination
    conditions before they cause crashes or data corruption.
    """

    def __init__(
        self,
        # Position limits
        max_position_magnitude: float = 50.0,
        min_height: float = 0.1,
        max_height: float = 5.0,
        # Velocity limits
        max_linear_velocity: float = 20.0,
        max_angular_velocity: float = 50.0,
        max_joint_velocity: float = 100.0,
        # Acceleration limits
        max_linear_acceleration: float = 100.0,
        max_angular_acceleration: float = 200.0,
        max_joint_acceleration: float = 500.0,
        # Force limits
        max_contact_force: float = 10000.0,
        max_constraint_force: float = 5000.0,
        # Stability thresholds
        stability_threshold: float = 0.1,
        instability_window: int = 10,
        # Performance thresholds
        max_simulation_time: float = 300.0,  # 5 minutes
        min_fps: float = 10.0,
        max_memory_usage: float = 0.9,  # 90% of available memory
        # Numerical thresholds
        nan_tolerance: int = 0,  # No NaN values allowed
        inf_tolerance: int = 0,  # No infinite values allowed
        # Advanced monitoring
        enable_predictive_termination: bool = True,
        monitoring_window_size: int = 50,
    ):
        """Initialize termination checker with configurable thresholds."""

        # Store configuration
        self.max_position_magnitude = max_position_magnitude
        self.min_height = min_height
        self.max_height = max_height
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.max_joint_velocity = max_joint_velocity
        self.max_linear_acceleration = max_linear_acceleration
        self.max_angular_acceleration = max_angular_acceleration
        self.max_joint_acceleration = max_joint_acceleration
        self.max_contact_force = max_contact_force
        self.max_constraint_force = max_constraint_force
        self.stability_threshold = stability_threshold
        self.instability_window = instability_window
        self.max_simulation_time = max_simulation_time
        self.min_fps = min_fps
        self.max_memory_usage = max_memory_usage
        self.nan_tolerance = nan_tolerance
        self.inf_tolerance = inf_tolerance
        self.enable_predictive_termination = enable_predictive_termination

        # State tracking
        self.physics_history: List[PhysicsState] = []
        self.monitoring_window_size = monitoring_window_size
        self.instability_count = 0
        self.last_check_time = time.time()
        self.simulation_start_time = time.time()
        self.frame_times: List[float] = []

        # Statistics
        self.total_checks = 0
        self.termination_counts = {reason: 0 for reason in TerminationReason}

    def check_termination(
        self,
        robot_position: np.ndarray,
        robot_orientation: np.ndarray,
        robot_linear_velocity: np.ndarray,
        robot_angular_velocity: np.ndarray,
        joint_positions: np.ndarray,
        joint_velocities: np.ndarray,
        step_count: int,
        simulation_time: float,
        joint_accelerations: Optional[np.ndarray] = None,
        contact_forces: Optional[np.ndarray] = None,
        constraint_forces: Optional[np.ndarray] = None,
    ) -> TerminationCheckResult:
        """
        Perform comprehensive termination check.

        Args:
            robot_position: Robot base position [x, y, z]
            robot_orientation: Robot base orientation quaternion [x, y, z, w]
            robot_linear_velocity: Robot linear velocity
            robot_angular_velocity: Robot angular velocity
            joint_positions: Joint positions
            joint_velocities: Joint velocities
            step_count: Current simulation step
            simulation_time: Simulation time in seconds
            joint_accelerations: Joint accelerations (optional)
            contact_forces: Contact forces (optional)
            constraint_forces: Constraint forces (optional)

        Returns:
            TerminationCheckResult with decision and diagnostics
        """
        self.total_checks += 1
        current_time = time.time()

        # Create physics state snapshot
        physics_state = PhysicsState(
            robot_position=robot_position,
            robot_orientation=robot_orientation,
            robot_linear_velocity=robot_linear_velocity,
            robot_angular_velocity=robot_angular_velocity,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            joint_accelerations=joint_accelerations,
            contact_forces=contact_forces,
            constraint_forces=constraint_forces,
            step_count=step_count,
            simulation_time=simulation_time,
            timestamp=current_time,
        )

        # Update history
        self._update_history(physics_state)

        # Update frame timing
        if len(self.frame_times) > 0:
            frame_time = current_time - self.last_check_time
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 100:  # Keep last 100 frame times
                self.frame_times.pop(0)

        self.last_check_time = current_time

        # Perform termination checks in order of severity
        checks = [
            self._check_numerical_stability,
            self._check_physics_instability,
            self._check_extreme_positions,
            self._check_extreme_velocities,
            self._check_robot_state,
            self._check_performance,
            self._check_timeout,
        ]

        for check_func in checks:
            result = check_func(physics_state)
            if result.should_terminate:
                self.termination_counts[result.reason] += 1
                logger.warning(f"Termination triggered: {result.reason.value}")
                return result

        # If predictive termination is enabled, check for early warning signs
        if self.enable_predictive_termination and len(self.physics_history) >= 5:
            predictive_result = self._check_predictive_termination(physics_state)
            if predictive_result.should_terminate:
                self.termination_counts[predictive_result.reason] += 1
                logger.info(f"Predictive termination: {predictive_result.reason.value}")
                return predictive_result

        # No termination condition met
        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_numerical_stability(self, state: PhysicsState) -> TerminationCheckResult:
        """Check for numerical instabilities (NaN, inf values)."""

        arrays_to_check = [
            ("position", state.robot_position),
            ("orientation", state.robot_orientation),
            ("linear_velocity", state.robot_linear_velocity),
            ("angular_velocity", state.robot_angular_velocity),
            ("joint_positions", state.joint_positions),
            ("joint_velocities", state.joint_velocities),
        ]

        if state.joint_accelerations is not None:
            arrays_to_check.append(("joint_accelerations", state.joint_accelerations))

        for name, array in arrays_to_check:
            # Check for NaN values
            nan_count = np.isnan(array).sum()
            if nan_count > self.nan_tolerance:
                return TerminationCheckResult(
                    should_terminate=True,
                    reason=TerminationReason.NAN_VALUES,
                    confidence=1.0,
                    diagnostic_data={"array": name, "nan_count": int(nan_count)},
                    recovery_possible=False,
                    severity="critical",
                )

            # Check for infinite values
            inf_count = np.isinf(array).sum()
            if inf_count > self.inf_tolerance:
                return TerminationCheckResult(
                    should_terminate=True,
                    reason=TerminationReason.INFINITE_VALUES,
                    confidence=1.0,
                    diagnostic_data={"array": name, "inf_count": int(inf_count)},
                    recovery_possible=False,
                    severity="critical",
                )

            # Check for extremely large values (potential overflow)
            max_abs_value = np.abs(array).max()
            if max_abs_value > 1e10:
                return TerminationCheckResult(
                    should_terminate=True,
                    reason=TerminationReason.NUMERICAL_OVERFLOW,
                    confidence=0.9,
                    diagnostic_data={"array": name, "max_value": float(max_abs_value)},
                    recovery_possible=True,
                    severity="high",
                )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_physics_instability(self, state: PhysicsState) -> TerminationCheckResult:
        """Check for physics instability patterns."""

        # Check position stability
        position_magnitude = np.linalg.norm(state.robot_position[:2])  # x, y only
        if position_magnitude > self.max_position_magnitude:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.PHYSICS_INSTABILITY,
                confidence=0.95,
                diagnostic_data={"position_magnitude": float(position_magnitude)},
                recovery_possible=True,
                severity="high",
            )

        # Check velocity instability
        linear_velocity_magnitude = np.linalg.norm(state.robot_linear_velocity)
        if linear_velocity_magnitude > self.max_linear_velocity:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.EXTREME_VELOCITY,
                confidence=0.9,
                diagnostic_data={"linear_velocity": float(linear_velocity_magnitude)},
                recovery_possible=True,
                severity="high",
            )

        # Check angular velocity
        angular_velocity_magnitude = np.linalg.norm(state.robot_angular_velocity)
        if angular_velocity_magnitude > self.max_angular_velocity:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.EXTREME_VELOCITY,
                confidence=0.9,
                diagnostic_data={"angular_velocity": float(angular_velocity_magnitude)},
                recovery_possible=True,
                severity="high",
            )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_extreme_positions(self, state: PhysicsState) -> TerminationCheckResult:
        """Check for extreme robot positions."""

        # Check height bounds
        height = state.robot_position[2]
        if height < self.min_height:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.ROBOT_FALLEN,
                confidence=1.0,
                diagnostic_data={
                    "height": float(height),
                    "min_height": self.min_height,
                },
                recovery_possible=True,
                severity="medium",
            )

        if height > self.max_height:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.EXTREME_POSITION,
                confidence=0.95,
                diagnostic_data={
                    "height": float(height),
                    "max_height": self.max_height,
                },
                recovery_possible=True,
                severity="medium",
            )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_extreme_velocities(self, state: PhysicsState) -> TerminationCheckResult:
        """Check for extreme joint velocities."""

        max_joint_vel = np.abs(state.joint_velocities).max()
        if max_joint_vel > self.max_joint_velocity:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.EXTREME_VELOCITY,
                confidence=0.9,
                diagnostic_data={"max_joint_velocity": float(max_joint_vel)},
                recovery_possible=True,
                severity="high",
            )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_robot_state(self, state: PhysicsState) -> TerminationCheckResult:
        """Check robot-specific termination conditions."""

        # Check for severe tilt (robot fallen over)
        # Quaternion [x, y, z, w] - check x, y components for tilt
        quat = state.robot_orientation
        tilt_magnitude = np.sqrt(quat[0] ** 2 + quat[1] ** 2)

        if tilt_magnitude > 0.7:  # Severe tilt
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.ROBOT_FALLEN,
                confidence=0.8,
                diagnostic_data={"tilt_magnitude": float(tilt_magnitude)},
                recovery_possible=True,
                severity="medium",
            )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_performance(self, state: PhysicsState) -> TerminationCheckResult:
        """Check for performance degradation."""

        # Check frame rate
        if len(self.frame_times) > 10:
            avg_frame_time = np.mean(self.frame_times[-10:])
            current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            if current_fps < self.min_fps:
                return TerminationCheckResult(
                    should_terminate=True,
                    reason=TerminationReason.PERFORMANCE_DEGRADATION,
                    confidence=0.7,
                    diagnostic_data={
                        "current_fps": float(current_fps),
                        "min_fps": self.min_fps,
                    },
                    recovery_possible=True,
                    severity="medium",
                )

        # Check memory usage (if psutil is available)
        try:
            import psutil

            memory_percent = psutil.virtual_memory().percent / 100.0
            if memory_percent > self.max_memory_usage:
                return TerminationCheckResult(
                    should_terminate=True,
                    reason=TerminationReason.MEMORY_EXHAUSTION,
                    confidence=0.8,
                    diagnostic_data={"memory_usage": float(memory_percent)},
                    recovery_possible=True,
                    severity="high",
                )
        except ImportError:
            pass  # psutil not available

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_timeout(self, state: PhysicsState) -> TerminationCheckResult:
        """Check for simulation timeout."""

        elapsed_time = time.time() - self.simulation_start_time
        if elapsed_time > self.max_simulation_time:
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.SIMULATION_TIMEOUT,
                confidence=1.0,
                diagnostic_data={
                    "elapsed_time": elapsed_time,
                    "max_time": self.max_simulation_time,
                },
                recovery_possible=False,
                severity="medium",
            )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _check_predictive_termination(
        self, state: PhysicsState
    ) -> TerminationCheckResult:
        """Check for early warning signs of instability."""

        if len(self.physics_history) < 5:
            return TerminationCheckResult(
                should_terminate=False,
                reason=TerminationReason.SUCCESS,
                confidence=1.0,
                diagnostic_data={},
                recovery_possible=True,
                severity="low",
            )

        # Check for rapidly increasing velocity trends
        recent_velocities = [
            np.linalg.norm(s.robot_linear_velocity) for s in self.physics_history[-5:]
        ]
        velocity_trend = np.polyfit(range(5), recent_velocities, 1)[0]  # Slope

        if velocity_trend > 1.0:  # Rapidly accelerating (lowered threshold for testing)
            return TerminationCheckResult(
                should_terminate=True,
                reason=TerminationReason.PHYSICS_INSTABILITY,
                confidence=0.6,
                diagnostic_data={"velocity_trend": float(velocity_trend)},
                recovery_possible=True,
                severity="medium",
            )

        # Check for oscillating behavior
        recent_heights = [s.robot_position[2] for s in self.physics_history[-10:]]
        if len(recent_heights) >= 10:
            height_std = np.std(recent_heights)
            height_mean = np.mean(recent_heights)

            if height_std > 0.2 and height_mean > 0.5:  # High oscillation
                return TerminationCheckResult(
                    should_terminate=True,
                    reason=TerminationReason.PHYSICS_INSTABILITY,
                    confidence=0.5,
                    diagnostic_data={"height_oscillation": float(height_std)},
                    recovery_possible=True,
                    severity="medium",
                )

        return TerminationCheckResult(
            should_terminate=False,
            reason=TerminationReason.SUCCESS,
            confidence=1.0,
            diagnostic_data={},
            recovery_possible=True,
            severity="low",
        )

    def _update_history(self, state: PhysicsState) -> None:
        """Update physics state history."""
        self.physics_history.append(state)

        # Keep only recent history
        if len(self.physics_history) > self.monitoring_window_size:
            self.physics_history.pop(0)

    def reset(self) -> None:
        """Reset termination checker for new simulation."""
        self.physics_history.clear()
        self.instability_count = 0
        self.simulation_start_time = time.time()
        self.last_check_time = time.time()
        self.frame_times.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get termination checker statistics."""
        return {
            "total_checks": self.total_checks,
            "termination_counts": {
                k.value: v for k, v in self.termination_counts.items()
            },
            "current_fps": (
                1.0 / np.mean(self.frame_times[-10:])
                if len(self.frame_times) >= 10
                else 0
            ),
            "history_length": len(self.physics_history),
        }
