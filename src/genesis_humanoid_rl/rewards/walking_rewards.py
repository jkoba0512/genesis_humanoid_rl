"""
Walking-specific reward functions for humanoid training.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple

from ..infrastructure.adapters.tensor_adapter import (
    safe_sqrt,
    safe_sum,
    safe_mean,
    safe_clip,
)


class WalkingRewardFunction:
    """
    Comprehensive reward function for humanoid walking.

    Encourages forward motion while maintaining stability and energy efficiency.
    """

    def __init__(
        self,
        velocity_weight: float = 1.0,
        stability_weight: float = 0.5,
        height_weight: float = 0.3,
        energy_weight: float = -0.1,
        smoothness_weight: float = -0.1,
        height_safety_weight: float = -0.5,
        target_velocity: float = 1.0,
        target_height: float = 0.8,
        max_height: float = 1.2,
        min_height: float = 0.3,
    ):
        """Initialize reward function with weights and targets."""
        self.velocity_weight = velocity_weight
        self.stability_weight = stability_weight
        self.height_weight = height_weight
        self.energy_weight = energy_weight
        self.smoothness_weight = smoothness_weight
        self.height_safety_weight = height_safety_weight

        self.target_velocity = target_velocity
        self.target_height = target_height
        self.max_height = max_height
        self.min_height = min_height

        self.previous_action = None

        # Weights dictionary for external access
        self.weights = {
            "velocity": velocity_weight,
            "stability": stability_weight,
            "height": height_weight,
            "energy": energy_weight,
            "smoothness": smoothness_weight,
        }

    def compute_reward(
        self,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
        joint_positions: torch.Tensor,
        joint_velocities: torch.Tensor,
        action: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute comprehensive walking reward.

        Args:
            robot_pos: Robot base position [x, y, z]
            robot_quat: Robot base quaternion [w, x, y, z]
            joint_positions: Joint positions
            joint_velocities: Joint velocities
            action: Current action taken

        Returns:
            total_reward: Combined reward value
            reward_components: Breakdown of reward components
        """
        components = {}

        # 1. Forward velocity reward
        forward_velocity = (
            robot_pos[0] if len(robot_pos.shape) == 1 else robot_pos[0, 0]
        )
        velocity_reward = min(forward_velocity / self.target_velocity, 2.0)
        components["velocity_reward"] = self.velocity_weight * velocity_reward

        # 2. Stability reward (upright posture)
        if len(robot_quat.shape) == 1:
            quat = robot_quat
        else:
            quat = robot_quat[0]

        # Use x,y components of quaternion to measure tilt
        # quat is [x, y, z, w] where x,y represent tilt
        tilt_magnitude = safe_sqrt(quat[0] ** 2 + quat[1] ** 2)
        if hasattr(tilt_magnitude, "item"):
            tilt_magnitude = tilt_magnitude.item()
        stability_reward = max(0.0, 1.0 - 2.0 * float(tilt_magnitude))
        components["stability_reward"] = self.stability_weight * stability_reward

        # 3. Height maintenance reward
        current_height = robot_pos[2] if len(robot_pos.shape) == 1 else robot_pos[0, 2]
        height_difference = abs(current_height.item() - self.target_height)
        height_reward = max(0.0, 1.0 - height_difference)
        components["height_reward"] = self.height_weight * height_reward

        # 4. Energy efficiency penalty
        if isinstance(joint_velocities, torch.Tensor):
            joint_vel_np = joint_velocities.cpu().numpy()
        else:
            joint_vel_np = joint_velocities

        energy_penalty = min(np.mean(joint_vel_np**2), 10.0)
        components["energy_penalty"] = self.energy_weight * energy_penalty

        # 5. Action smoothness penalty
        if self.previous_action is not None:
            action_diff = action - self.previous_action
            smoothness_penalty = np.mean(action_diff**2)
            components["smoothness_penalty"] = (
                self.smoothness_weight * smoothness_penalty
            )
        else:
            components["smoothness_penalty"] = 0.0

        # 6. Height safety penalty
        current_height_val = (
            current_height.item() if hasattr(current_height, "item") else current_height
        )
        if current_height_val > self.max_height:
            height_safety_penalty = (current_height_val - self.max_height) * 2.0
            components["height_safety"] = (
                self.height_safety_weight * height_safety_penalty
            )
        else:
            components["height_safety"] = 0.0

        # Store action for next iteration
        self.previous_action = action.copy()

        # Calculate total reward
        total_reward = sum(components.values())

        return total_reward, components

    def update_weights(self, weights: Dict[str, float]) -> None:
        """Update reward component weights."""
        for key, value in weights.items():
            if key == "velocity":
                self.velocity_weight = value
            elif key == "stability":
                self.stability_weight = value
            elif key == "height":
                self.height_weight = value
            elif key == "energy":
                self.energy_weight = value
            elif key == "smoothness":
                self.smoothness_weight = value

        # Update weights dictionary
        self.weights.update(weights)

    def reset(self) -> None:
        """Reset previous state tracking."""
        self.previous_action = None
        # For the test that sets these attributes dynamically
        if hasattr(self, "_previous_pos"):
            self._previous_pos = None
        if hasattr(self, "_previous_action"):
            self._previous_action = None

    def should_terminate(
        self,
        robot_pos: torch.Tensor,
        robot_quat: torch.Tensor,
    ) -> Tuple[bool, str]:
        """
        Check if episode should terminate due to failure conditions.

        Returns:
            should_terminate: Whether to end episode
            reason: Termination reason
        """
        current_height = robot_pos[2] if len(robot_pos.shape) == 1 else robot_pos[0, 2]
        current_height_val = (
            current_height.item() if hasattr(current_height, "item") else current_height
        )

        # Check height bounds
        if current_height_val < self.min_height:
            return True, f"Robot fell (height: {current_height_val:.2f}m)"

        if current_height_val > 2.0:  # Very high jump
            return True, f"Robot jumped too high (height: {current_height_val:.2f}m)"

        # Check position bounds
        x_pos = robot_pos[0] if len(robot_pos.shape) == 1 else robot_pos[0, 0]
        y_pos = robot_pos[1] if len(robot_pos.shape) == 1 else robot_pos[0, 1]

        x_val = x_pos.item() if hasattr(x_pos, "item") else x_pos
        y_val = y_pos.item() if hasattr(y_pos, "item") else y_pos

        if abs(x_val) > 10.0:
            return True, f"Robot moved too far in X (x: {x_val:.2f}m)"

        if abs(y_val) > 5.0:
            return True, f"Robot moved too far in Y (y: {y_val:.2f}m)"

        return False, ""
