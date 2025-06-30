"""
Simulation-specific exceptions for humanoid robotics training.

Provides domain-specific error handling for training scenarios,
physics simulation, and robot control operations.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .genesis_exceptions import GenesisError, GenesisErrorType

logger = logging.getLogger(__name__)


class SimulationError(Exception):
    """Base exception for simulation-related errors."""

    def __init__(
        self,
        message: str,
        recovery_action: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.recovery_action = recovery_action
        self.context = context or {}


class PhysicsInstabilityError(SimulationError):
    """
    Error indicating physics simulation has become unstable.

    This typically occurs when forces become extreme, joints violate limits,
    or numerical errors accumulate beyond recoverable bounds.
    """

    def __init__(
        self,
        message: str,
        robot_state: Optional[Dict[str, Any]] = None,
        simulation_step: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            recovery_action="Reset simulation with adjusted parameters",
            **kwargs,
        )
        self.robot_state = robot_state
        self.simulation_step = simulation_step


class RobotControlError(SimulationError):
    """
    Error in robot control or action execution.

    Occurs when action commands are invalid, joint limits are exceeded,
    or robot becomes uncontrollable.
    """

    def __init__(
        self,
        message: str,
        action: Optional[Any] = None,
        joint_states: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(
            message, recovery_action="Validate and clip action commands", **kwargs
        )
        self.action = action
        self.joint_states = joint_states


class EnvironmentError(SimulationError):
    """
    Error in environment setup or state management.

    Occurs when environment configuration is invalid or
    environment state becomes corrupted.
    """

    def __init__(
        self,
        message: str,
        environment_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(
            message,
            recovery_action="Reinitialize environment with valid configuration",
            **kwargs,
        )
        self.environment_config = environment_config


class TrainingError(SimulationError):
    """
    Error during training process.

    Occurs when training encounters unrecoverable issues
    or curriculum progression fails.
    """

    def __init__(
        self,
        message: str,
        episode_data: Optional[Dict[str, Any]] = None,
        training_metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(
            message, recovery_action="Adjust training parameters and restart", **kwargs
        )
        self.episode_data = episode_data
        self.training_metrics = training_metrics


@dataclass
class SimulationErrorContext:
    """Context information for simulation error analysis."""

    error_type: str
    simulation_step: int
    episode_number: int
    robot_position: tuple[float, float, float]
    robot_orientation: tuple[float, float, float, float]
    joint_positions: list[float]
    joint_velocities: list[float]
    last_action: list[float]
    physics_timestep: float
    total_reward: float
    episode_length: int


class SimulationErrorHandler:
    """Handles simulation errors with automatic recovery strategies."""

    def __init__(self):
        self.error_count = 0
        self.recovery_attempts = {}
        self.max_recovery_attempts = 3

    def handle_error(
        self, error: Exception, context: Optional[SimulationErrorContext] = None
    ) -> Dict[str, Any]:
        """
        Handle simulation error with appropriate recovery strategy.

        Args:
            error: The simulation error that occurred
            context: Additional context about the simulation state

        Returns:
            Recovery action plan with suggested parameters
        """
        self.error_count += 1

        # Log error with context
        self._log_error(error, context)

        # Determine recovery strategy
        if isinstance(error, PhysicsInstabilityError):
            return self._handle_physics_instability(error, context)
        elif isinstance(error, RobotControlError):
            return self._handle_robot_control_error(error, context)
        elif isinstance(error, EnvironmentError):
            return self._handle_environment_error(error, context)
        elif isinstance(error, TrainingError):
            return self._handle_training_error(error, context)
        elif isinstance(error, GenesisError):
            return self._handle_genesis_error(error, context)
        else:
            return self._handle_unknown_error(error, context)

    def _handle_physics_instability(
        self, error: PhysicsInstabilityError, context: Optional[SimulationErrorContext]
    ) -> Dict[str, Any]:
        """Handle physics instability with parameter adjustment."""

        recovery_plan = {
            "action": "reset_simulation",
            "adjustments": {
                "physics_timestep": 0.8,  # Reduce timestep by 20%
                "solver_iterations": 1.5,  # Increase solver iterations
                "damping_factor": 1.2,  # Increase damping
                "action_scaling": 0.8,  # Reduce action magnitude
            },
            "retry_count": self.recovery_attempts.get("physics_instability", 0) + 1,
        }

        self.recovery_attempts["physics_instability"] = recovery_plan["retry_count"]

        # If too many attempts, suggest more drastic measures
        if recovery_plan["retry_count"] > self.max_recovery_attempts:
            recovery_plan.update(
                {
                    "action": "reset_episode",
                    "adjustments": {
                        "physics_timestep": 0.5,  # More aggressive timestep reduction
                        "enable_debug_mode": True,
                        "use_simple_geometry": True,
                    },
                }
            )

        return recovery_plan

    def _handle_robot_control_error(
        self, error: RobotControlError, context: Optional[SimulationErrorContext]
    ) -> Dict[str, Any]:
        """Handle robot control error with action validation."""

        return {
            "action": "validate_and_clip_actions",
            "adjustments": {
                "action_clipping": True,
                "joint_limit_padding": 0.05,  # 5% padding from joint limits
                "velocity_limit_scaling": 0.8,  # Reduce velocity limits
                "force_limit_scaling": 0.9,  # Reduce force limits
            },
            "retry_count": self.recovery_attempts.get("robot_control", 0) + 1,
        }

    def _handle_environment_error(
        self, error: EnvironmentError, context: Optional[SimulationErrorContext]
    ) -> Dict[str, Any]:
        """Handle environment error with reinitialization."""

        return {
            "action": "reinitialize_environment",
            "adjustments": {
                "reset_robot_pose": True,
                "clear_simulation_cache": True,
                "validate_configuration": True,
            },
            "retry_count": self.recovery_attempts.get("environment", 0) + 1,
        }

    def _handle_training_error(
        self, error: TrainingError, context: Optional[SimulationErrorContext]
    ) -> Dict[str, Any]:
        """Handle training error with curriculum adjustment."""

        return {
            "action": "adjust_training_parameters",
            "adjustments": {
                "reduce_difficulty": True,
                "extend_episode_length": True,
                "adjust_reward_scaling": 0.8,
                "enable_curriculum_fallback": True,
            },
            "retry_count": self.recovery_attempts.get("training", 0) + 1,
        }

    def _handle_genesis_error(
        self, error: GenesisError, context: Optional[SimulationErrorContext]
    ) -> Dict[str, Any]:
        """Handle Genesis-specific error using error context."""

        return {
            "action": "apply_genesis_recovery",
            "adjustments": {
                "use_recovery_suggestions": error.recovery_suggestions,
                "severity": error.severity,
                "retryable": error.retryable,
            },
            "genesis_error_type": error.error_type.value,
            "retry_count": self.recovery_attempts.get(
                f"genesis_{error.error_type.value}", 0
            )
            + 1,
        }

    def _handle_unknown_error(
        self, error: Exception, context: Optional[SimulationErrorContext]
    ) -> Dict[str, Any]:
        """Handle unknown error with conservative recovery."""

        return {
            "action": "conservative_recovery",
            "adjustments": {
                "reset_to_safe_state": True,
                "enable_debug_logging": True,
                "reduce_complexity": True,
            },
            "retry_count": self.recovery_attempts.get("unknown", 0) + 1,
        }

    def _log_error(
        self, error: Exception, context: Optional[SimulationErrorContext]
    ) -> None:
        """Log error with full context for debugging."""

        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "total_errors": self.error_count,
        }

        if context:
            error_info.update(
                {
                    "simulation_step": context.simulation_step,
                    "episode_number": context.episode_number,
                    "robot_position": context.robot_position,
                    "episode_length": context.episode_length,
                    "total_reward": context.total_reward,
                }
            )

        logger.error(f"Simulation error occurred: {error_info}")

    def reset_recovery_counts(self) -> None:
        """Reset recovery attempt counters."""
        self.recovery_attempts.clear()
        self.error_count = 0


def handle_simulation_error(
    error: Exception, context: Optional[SimulationErrorContext] = None
) -> Dict[str, Any]:
    """
    Convenience function to handle simulation errors.

    Args:
        error: The simulation error that occurred
        context: Additional context about the simulation state

    Returns:
        Recovery action plan
    """
    handler = SimulationErrorHandler()
    return handler.handle_error(error, context)
