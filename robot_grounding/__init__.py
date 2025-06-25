"""
Robot Grounding Library for Genesis

Provides automatic calculation of robot grounding height by analyzing 
URDF structure and detecting foot links for proper ground placement.
"""

__version__ = "0.1.0"
__author__ = "Genesis Humanoid Learning Project"

from .calculator import RobotGroundingCalculator
from .detector import FootDetector
from .utils import get_lowest_z_position, calculate_grounding_offset

__all__ = [
    'RobotGroundingCalculator',
    'FootDetector',
    'get_lowest_z_position',
    'calculate_grounding_offset'
]