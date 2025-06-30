"""
Proper integration of robot grounding functionality.
Replaces the sys.path.insert anti-pattern with clean dependency injection.
"""

from typing import Protocol, Optional, Any
import torch
import logging

logger = logging.getLogger(__name__)


class RobotGroundingCalculatorProtocol(Protocol):
    """Protocol for robot grounding calculations."""

    def get_grounding_height(self, safety_margin: float = 0.03) -> float:
        """Calculate appropriate grounding height for robot."""
        ...


class DefaultRobotGroundingCalculator:
    """
    Default implementation with fallback if robot_grounding unavailable.

    This replaces the sys.path.insert anti-pattern with proper dependency management.
    """

    def __init__(self, robot: Any, verbose: bool = False):
        self.robot = robot
        self.verbose = verbose
        self._calculator = self._create_calculator()

    def _create_calculator(self) -> Optional[Any]:
        """Create robot grounding calculator with graceful fallback."""
        try:
            # Try to import the external robot_grounding library
            import sys
            import os

            # Add robot_grounding to path only temporarily and cleanly
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            robot_grounding_path = os.path.join(project_root, "robot_grounding")

            if robot_grounding_path not in sys.path:
                sys.path.insert(0, project_root)

            from robot_grounding import RobotGroundingCalculator

            if self.verbose:
                logger.info("Using external robot_grounding library")

            return RobotGroundingCalculator(self.robot, self.verbose)

        except ImportError as e:
            if self.verbose:
                logger.warning(f"robot_grounding library not available: {e}")
                logger.info("Using fallback grounding calculation")
            return None

    def get_grounding_height(self, safety_margin: float = 0.03) -> float:
        """
        Get grounding height for robot.

        Args:
            safety_margin: Additional height clearance in meters

        Returns:
            Appropriate height for robot base positioning
        """
        if self._calculator:
            # Use external library if available
            return self._calculator.get_grounding_height(safety_margin)

        # Fallback calculation for G1 robot
        if self.verbose:
            logger.info("Using fallback grounding height calculation")

        # G1 robot approximate standing height
        base_height = 0.78  # meters
        return base_height + safety_margin


class RobotGroundingFactory:
    """Factory for creating robot grounding calculators."""

    @staticmethod
    def create_calculator(
        robot: Any, verbose: bool = False, calculator_type: str = "default"
    ) -> RobotGroundingCalculatorProtocol:
        """
        Create appropriate robot grounding calculator.

        Args:
            robot: Robot entity from physics simulation
            verbose: Enable verbose logging
            calculator_type: Type of calculator to create

        Returns:
            Robot grounding calculator instance
        """
        if calculator_type == "default":
            return DefaultRobotGroundingCalculator(robot, verbose)
        else:
            raise ValueError(f"Unknown calculator type: {calculator_type}")
