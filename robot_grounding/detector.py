"""
Foot link detection logic for various robot types.
"""

from typing import List, Dict, Any


class FootDetector:
    """
    Detect foot links in robot structures.
    """
    
    # Common foot-related keywords
    FOOT_KEYWORDS = [
        'foot', 'ankle', 'toe', 'sole', 'heel',
        'end_effector', 'ee', 'contact'
    ]
    
    # Keywords to exclude (usually hands)
    EXCLUDE_KEYWORDS = [
        'hand', 'finger', 'palm', 'wrist', 'gripper',
        'thumb', 'index', 'middle', 'ring', 'pinky'
    ]
    
    @staticmethod
    def is_end_link(link, all_links) -> bool:
        """
        Check if a link is an end link (no children).
        
        Args:
            link: Link to check
            all_links: List of all robot links
            
        Returns:
            True if link has no children
        """
        # Check if any other link has this link as parent
        if not hasattr(link, 'idx'):
            return False
            
        link_idx = link.idx
        for other_link in all_links:
            if hasattr(other_link, 'parent_idx') and other_link.parent_idx == link_idx:
                return False
        return True
    
    @staticmethod
    def is_foot_candidate(link_name: str) -> bool:
        """
        Check if link name suggests it's a foot based on keywords.
        
        Args:
            link_name: Name of the link
            
        Returns:
            True if name matches foot patterns
        """
        name_lower = link_name.lower()
        
        # Check exclusions first
        for exclude in FootDetector.EXCLUDE_KEYWORDS:
            if exclude in name_lower:
                return False
        
        # Check foot keywords
        for keyword in FootDetector.FOOT_KEYWORDS:
            if keyword in name_lower:
                return True
        
        return False
    
    @staticmethod
    def detect_foot_links(robot) -> List[Any]:
        """
        Detect foot links in a robot.
        
        Args:
            robot: Genesis robot entity
            
        Returns:
            List of detected foot links
        """
        foot_links = []
        all_links = robot.links
        
        # Find all end links
        end_links = []
        for link in all_links:
            if FootDetector.is_end_link(link, all_links):
                end_links.append(link)
        
        # Filter for foot candidates
        for link in end_links:
            if hasattr(link, 'name') and FootDetector.is_foot_candidate(link.name):
                foot_links.append(link)
        
        # If no foot links found by name, use heuristics
        if not foot_links and end_links:
            # For humanoids, typically look for lowest end links
            # This is a fallback strategy
            base_pos = robot.get_pos()
            
            # Sort end links by their typical position (will implement after testing)
            # For now, just return empty
            pass
        
        return foot_links