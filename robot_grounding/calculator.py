"""
Main calculator class for robot grounding height computation.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple

from .detector import FootDetector
from .utils import get_link_world_position, get_lowest_z_position, calculate_grounding_offset


class RobotGroundingCalculator:
    """
    Calculate appropriate grounding height for robots loaded in Genesis.
    
    This class analyzes robot structure to find foot links and compute
    the height needed to place the robot with feet on the ground.
    """
    
    def __init__(self, robot, verbose: bool = True):
        """
        Initialize the calculator with a Genesis robot entity.
        
        Args:
            robot: Genesis robot entity (from scene.add_entity)
            verbose: Whether to print debug information
        """
        self.robot = robot
        self.verbose = verbose
        
        # Get basic robot information
        self.n_links = robot.n_links
        self.n_dofs = robot.n_dofs
        
        if self.verbose:
            print(f"Initialized RobotGroundingCalculator")
            print(f"  Robot links: {self.n_links}")
            print(f"  Robot DOFs: {self.n_dofs}")
        
        # Find foot links (will be implemented next)
        self.foot_links = self._detect_foot_links()
        
    def _detect_foot_links(self) -> List:
        """
        Detect foot links by finding end effectors with specific patterns.
        
        Returns:
            List of foot link objects
        """
        if self.verbose:
            print("Detecting foot links...")
        
        # Use FootDetector to find foot links
        foot_links = FootDetector.detect_foot_links(self.robot)
        
        if self.verbose:
            print(f"  Found {len(foot_links)} foot links")
            for link in foot_links:
                if hasattr(link, 'name'):
                    print(f"    - {link.name}")
        
        return foot_links
    
    def get_grounding_height(self, safety_margin: float = 0.005) -> float:
        """
        Calculate the height to place robot base so feet touch the ground.
        
        Args:
            safety_margin: Small margin above ground to ensure contact (default 5mm)
        
        Returns:
            Height in meters for robot base placement
        """
        if self.verbose:
            print("Calculating grounding height...")
        
        # Check if we have foot links
        if not self.foot_links:
            if self.verbose:
                print("  Warning: No foot links detected, using default height")
            return 1.0  # Default fallback
        
        # Get current base position
        base_pos = self.robot.get_pos()
        if base_pos.dim() > 1:
            base_z = base_pos[0, 2].item()  # Batched position
        else:
            base_z = base_pos[2].item()  # Single position
        
        if self.verbose:
            print(f"  Current base height: {base_z:.3f}m")
        
        # Get lowest point of foot links
        foot_lowest_z = get_lowest_z_position(self.foot_links)
        
        if self.verbose:
            print(f"  Lowest foot point: {foot_lowest_z:.3f}m")
        
        # Calculate grounding height
        grounding_height = calculate_grounding_offset(
            base_z, foot_lowest_z, safety_margin
        )
        
        if self.verbose:
            print(f"  Calculated grounding height: {grounding_height:.3f}m")
            print(f"  (This will place feet {safety_margin*1000:.1f}mm above ground)")
        
        return grounding_height
    
    def get_current_foot_positions(self) -> Optional[torch.Tensor]:
        """
        Get current positions of detected foot links.
        
        Returns:
            Tensor of foot positions (n_feet, 3) or None if no feet detected
        """
        if not self.foot_links:
            return None
        
        positions = []
        for link in self.foot_links:
            try:
                pos = get_link_world_position(link)
                if pos.dim() > 1:
                    positions.append(pos[0])  # First environment if batched
                else:
                    positions.append(pos)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not get position for link: {e}")
        
        if positions:
            return torch.stack(positions)
        else:
            return None