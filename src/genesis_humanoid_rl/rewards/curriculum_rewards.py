"""
Curriculum-aware reward functions for humanoid walking.
Adapts reward components based on current curriculum stage.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional
from ..curriculum.curriculum_manager import CurriculumManager, CurriculumStage


class CurriculumRewardCalculator:
    """Calculates rewards based on current curriculum stage."""

    def __init__(self, curriculum_manager: CurriculumManager):
        self.curriculum = curriculum_manager
        self.previous_pos = None
        self.step_count = 0

    def calculate_reward(
        self, robot, action: np.ndarray, info: Dict[str, Any] = None
    ) -> float:
        """Calculate curriculum-aware reward."""
        if robot is None:
            return 0.0

        config = self.curriculum.get_current_config()
        weights = config.reward_weights

        # Get robot state
        pos = robot.get_pos().cpu().numpy()
        quat = robot.get_quat().cpu().numpy()
        joint_vel = robot.get_dofs_velocity().cpu().numpy()

        # Initialize previous position if needed
        if self.previous_pos is None:
            self.previous_pos = pos.copy()

        total_reward = 0.0
        reward_components = {}

        # 1. Stability Reward - keeping robot upright
        stability_reward = self._calculate_stability_reward(quat)
        total_reward += weights.get("stability", 0.0) * stability_reward
        reward_components["stability"] = stability_reward

        # 2. Height Maintenance - staying at proper walking height
        height_reward = self._calculate_height_reward(pos[2])
        total_reward += weights.get("height", 0.0) * height_reward
        reward_components["height"] = height_reward

        # 3. Velocity Reward - moving toward target speed
        velocity_reward = self._calculate_velocity_reward(pos)
        total_reward += weights.get("velocity", 0.0) * velocity_reward
        reward_components["velocity"] = velocity_reward

        # 4. Energy Efficiency - penalize excessive movement
        energy_penalty = self._calculate_energy_penalty(joint_vel)
        total_reward += weights.get("energy", 0.0) * energy_penalty
        reward_components["energy"] = energy_penalty

        # 5. Action Smoothness - encourage smooth control
        smoothness_penalty = self._calculate_smoothness_penalty(action, info)
        total_reward += weights.get("smoothness", 0.0) * smoothness_penalty
        reward_components["smoothness"] = smoothness_penalty

        # 6. Stage-specific rewards
        stage_reward = self._calculate_stage_specific_reward(pos, quat, action, info)
        total_reward += stage_reward
        reward_components["stage_specific"] = stage_reward

        # Update state
        self.previous_pos = pos.copy()
        self.step_count += 1

        return total_reward

    def _calculate_stability_reward(self, quat: np.ndarray) -> float:
        """Reward for maintaining upright posture."""
        # Quaternion tilt magnitude (x,y components indicate tilt)
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            tilt = np.sqrt(quat[0] ** 2 + quat[1] ** 2)
            stability = max(0.0, 1.0 - 2.0 * tilt)
        else:
            stability = 0.0

        return stability

    def _calculate_height_reward(self, height: float) -> float:
        """Reward for maintaining proper walking height."""
        target_height = 0.8  # Target height for G1 robot
        height_diff = abs(height - target_height)
        height_reward = max(0.0, 1.0 - height_diff)
        return height_reward

    def _calculate_velocity_reward(self, pos: np.ndarray) -> float:
        """Reward for moving at target velocity."""
        if self.previous_pos is None:
            return 0.0

        # Calculate forward velocity
        dt = 1.0 / 20.0  # Assuming 20 Hz control frequency
        forward_velocity = (pos[0] - self.previous_pos[0]) / dt

        # Get target velocity from curriculum
        target_velocity = self.curriculum.get_adaptive_target_velocity()

        if target_velocity == 0.0:
            # Balance stage - reward staying in place
            return max(0.0, 1.0 - abs(forward_velocity))
        else:
            # Walking stages - reward matching target velocity
            velocity_error = abs(forward_velocity - target_velocity)
            return max(0.0, 1.0 - velocity_error / target_velocity)

    def _calculate_energy_penalty(self, joint_velocities: np.ndarray) -> float:
        """Penalty for excessive joint movement."""
        energy = np.mean(np.square(joint_velocities))
        return -min(energy, 10.0)  # Cap penalty

    def _calculate_smoothness_penalty(
        self, action: np.ndarray, info: Dict[str, Any]
    ) -> float:
        """Penalty for jerky, non-smooth actions."""
        if info is None or "previous_action" not in info:
            return 0.0

        prev_action = info["previous_action"]
        if prev_action is None:
            return 0.0

        action_diff = np.mean(np.square(action - prev_action))
        return -min(action_diff, 2.0)  # Cap penalty

    def _calculate_stage_specific_reward(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        action: np.ndarray,
        info: Dict[str, Any],
    ) -> float:
        """Calculate rewards specific to current curriculum stage."""
        stage = self.curriculum.current_stage

        if stage == CurriculumStage.BALANCE:
            return self._balance_stage_reward(pos, quat)
        elif stage == CurriculumStage.SMALL_STEPS:
            return self._small_steps_reward(pos)
        elif stage == CurriculumStage.WALKING:
            return self._walking_reward(pos)
        elif stage == CurriculumStage.SPEED_CONTROL:
            return self._speed_control_reward(pos)
        elif stage == CurriculumStage.TURNING:
            return self._turning_reward(pos, info)
        else:
            return 0.0

    def _balance_stage_reward(self, pos: np.ndarray, quat: np.ndarray) -> float:
        """Specific rewards for balance training stage."""
        # Extra reward for minimal movement in X,Y
        movement_penalty = -(abs(pos[0]) + abs(pos[1])) * 0.1

        # Bonus for staying exactly upright
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            uprightness = 1.0 - np.sqrt(quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2)
            upright_bonus = max(0.0, uprightness) * 0.2
        else:
            upright_bonus = 0.0

        return movement_penalty + upright_bonus

    def _small_steps_reward(self, pos: np.ndarray) -> float:
        """Rewards for small steps stage."""
        # Bonus for small forward progress
        if self.previous_pos is not None:
            forward_progress = pos[0] - self.previous_pos[0]
            if 0 < forward_progress < 0.01:  # Very small steps
                return 0.1
        return 0.0

    def _walking_reward(self, pos: np.ndarray) -> float:
        """Rewards for basic walking stage."""
        # Bonus for consistent forward motion
        if self.previous_pos is not None:
            forward_progress = pos[0] - self.previous_pos[0]
            if forward_progress > 0:
                return min(forward_progress * 2.0, 0.2)  # Cap bonus
        return 0.0

    def _speed_control_reward(self, pos: np.ndarray) -> float:
        """Rewards for speed control stage."""
        # Additional reward for velocity precision
        if self.previous_pos is not None:
            dt = 1.0 / 20.0
            actual_velocity = (pos[0] - self.previous_pos[0]) / dt
            target_velocity = self.curriculum.get_adaptive_target_velocity()

            velocity_precision = 1.0 - abs(actual_velocity - target_velocity) / max(
                target_velocity, 0.1
            )
            return max(0.0, velocity_precision) * 0.3
        return 0.0

    def _turning_reward(self, pos: np.ndarray, info: Dict[str, Any]) -> float:
        """Rewards for turning control stage."""
        # This would require additional state tracking for target directions
        # For now, return 0 - implement when adding directional control
        return 0.0

    def reset_episode(self):
        """Reset episode-specific state."""
        self.previous_pos = None
        self.step_count = 0

    def is_terminated(self, robot) -> bool:
        """Check if episode should terminate based on curriculum stage."""
        if robot is None:
            return True

        config = self.curriculum.get_current_config()
        termination = config.termination_conditions

        pos = robot.get_pos().cpu().numpy()
        quat = robot.get_quat().cpu().numpy()

        # Height termination
        min_height = termination.get("min_height", 0.3)
        if pos[2] < min_height:
            return True

        # Tilt termination
        max_tilt = termination.get("max_tilt", 0.8)
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            tilt = np.sqrt(quat[0] ** 2 + quat[1] ** 2)
            if tilt > max_tilt:
                return True

        # Out of bounds
        if abs(pos[0]) > 10.0 or abs(pos[1]) > 5.0:
            return True

        # Height ceiling (prevent jumping)
        if pos[2] > 2.0:
            return True

        return False
