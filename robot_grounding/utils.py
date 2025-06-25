"""
Utility functions for robot grounding calculations.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def get_link_world_position(link) -> torch.Tensor:
    """
    Get the world position of a link.
    
    Args:
        link: Genesis link object
        
    Returns:
        Position tensor (3,) in world coordinates
    """
    if hasattr(link, 'get_pos'):
        return link.get_pos()
    else:
        raise AttributeError(f"Link does not have get_pos method")


def get_lowest_z_position(links: List) -> float:
    """
    Find the lowest Z position among a list of links.
    
    Args:
        links: List of Genesis link objects
        
    Returns:
        Lowest Z coordinate value
    """
    if not links:
        return 0.0
    
    lowest_z = float('inf')
    
    for link in links:
        try:
            pos = get_link_world_position(link)
            # Handle both single position and batched positions
            if pos.dim() == 1:
                z = pos[2].item()
            else:
                # For batched, take first environment
                z = pos[0, 2].item()
            
            if z < lowest_z:
                lowest_z = z
                
        except Exception as e:
            print(f"Warning: Could not get position for link: {e}")
            continue
    
    return lowest_z if lowest_z != float('inf') else 0.0


def calculate_grounding_offset(robot_base_z: float, foot_links_lowest_z: float, 
                             safety_margin: float = 0.005) -> float:
    """
    Calculate the offset needed to ground the robot.
    
    Args:
        robot_base_z: Current Z position of robot base
        foot_links_lowest_z: Lowest Z position of foot links
        safety_margin: Small margin to ensure contact (default 5mm)
        
    Returns:
        Height offset to place robot on ground
    """
    # The offset is the distance from foot to ground plus safety margin
    current_foot_height = foot_links_lowest_z
    desired_foot_height = safety_margin
    
    # Calculate how much to move the entire robot down
    offset = current_foot_height - desired_foot_height
    
    # Return the new base height
    return robot_base_z - offset