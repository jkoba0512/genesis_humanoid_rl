"""
Humanoid walking environment using Genesis physics engine.
Integrates with genesis_humanoid_learning library for Unitree G1 robot.
"""

import numpy as np
import genesis as gs
from gymnasium import Env, spaces
from typing import Any, Dict, Optional, Tuple
import os
import sys
import torch

# Add robot_grounding to Python path
# __file__ is in src/genesis_humanoid_rl/environments/
# We need to go up 3 levels to reach project root
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)
from robot_grounding import RobotGroundingCalculator


class HumanoidWalkingEnv(Env):
    """
    Humanoid walking environment for reinforcement learning.

    Uses Genesis physics engine with Unitree G1 humanoid robot.
    Integrates robot_grounding library for automatic positioning.
    """

    def __init__(
        self,
        render_mode: Optional[str] = None,
        simulation_fps: int = 100,
        control_freq: int = 20,
        episode_length: int = 1000,
        target_velocity: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.control_freq = control_freq
        self.episode_length = episode_length
        self.target_velocity = target_velocity

        # Initialize Genesis only if not already initialized
        try:
            gs.init()
        except RuntimeError as e:
            if "Genesis already initialized" not in str(e):
                raise

        # Scene and robot will be initialized in reset()
        self.scene = None
        self.robot = None
        self.current_step = 0

        # Action space will be defined after robot is loaded
        self.num_joints = None
        self.joint_names = None
        self.action_space = None  # Will be set in reset()

        # Observation space will be defined after robot is loaded
        self.observation_space = None  # Will be set in reset()

        self.previous_action = None  # Will be initialized in reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Create new scene
        scene_kwargs = {
            "sim_options": gs.options.SimOptions(
                dt=1.0 / self.simulation_fps,
                substeps=10,
            ),
            "show_viewer": False,  # Disable viewer for faster initialization
        }

        if self.render_mode is not None:
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                res=(1024, 768),
                max_FPS=self.simulation_fps,
            )

        self.scene = gs.Scene(**scene_kwargs)

        # Load ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Load Unitree G1 robot
        urdf_path = os.path.join(project_root, "assets/robots/g1/g1_29dof.urdf")

        # Load robot at initial height (will be adjusted by grounding calculator)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=(0, 0, 1.0),  # Initial position, will be adjusted
                euler=(0, 0, 0),
            ),
        )

        # Build scene first
        self.scene.build()

        # Apply automatic grounding
        if self.robot is not None:
            calculator = RobotGroundingCalculator(self.robot, verbose=True)
            grounding_height = calculator.get_grounding_height(safety_margin=0.03)
            self.robot.set_pos(torch.tensor([0, 0, grounding_height]))

            # Let the robot settle
            for _ in range(10):
                self.scene.step()

        # Get joint information after robot is loaded
        if self.robot is not None:
            self.num_joints = self.robot.n_dofs
            print(f"Loaded G1 robot with {self.num_joints} DOFs")
            print(f"Robot has {self.robot.n_links} links")

            # Update action space based on actual joint count
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
            )

            # Update observation space based on actual joint count
            # Position (3) + orientation (4) + joint positions + joint velocities
            # + previous action + target velocity (1)
            obs_dim = 3 + 4 + self.num_joints + self.num_joints + self.num_joints + 1
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

        self.current_step = 0
        self.previous_action = np.zeros(self.num_joints)

        # Initialize previous position for velocity calculation
        if self.robot is not None:
            self._previous_pos = self.robot.get_pos().cpu().numpy()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply action to robot joints
        if self.robot is not None:
            # Scale action to actual joint torque/position limits
            # For now, we'll use position control with the action as target positions
            # Scale from [-1, 1] to actual joint range (this is a simplification)
            current_pos = self.robot.get_dofs_position()

            # Simple position control: action represents desired position change
            target_pos = current_pos + torch.tensor(
                action * 0.1, dtype=current_pos.dtype, device=current_pos.device
            )  # Small position changes
            self.robot.control_dofs_position(target_pos)

        # Step simulation
        steps_per_control = self.simulation_fps // self.control_freq
        for _ in range(steps_per_control):
            self.scene.step()

        # Get observation
        obs = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward(action)

        # Update step counter
        self.current_step += 1

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.episode_length

        # Store previous action
        self.previous_action = action.copy()

        info = {
            "step": self.current_step,
            "terminated": terminated,
            "truncated": truncated,
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation from the environment."""
        if self.robot is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get robot base position and orientation
        pos = self.robot.get_pos().cpu().numpy()  # (3,)
        quat = self.robot.get_quat().cpu().numpy()  # (4,) quaternion (x, y, z, w)

        # Get joint positions and velocities
        joint_pos = self.robot.get_dofs_position().cpu().numpy()  # (num_joints,)
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()  # (num_joints,)

        # Concatenate all observations
        obs = np.concatenate(
            [
                pos,  # Base position (3)
                quat,  # Base orientation (4)
                joint_pos,  # Joint positions
                joint_vel,  # Joint velocities
                self.previous_action,  # Previous action
                [self.target_velocity],  # Target velocity (1)
            ]
        )

        return obs.astype(np.float32)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for the current state and action."""
        if self.robot is None:
            return 0.0

        total_reward = 0.0

        # Get current robot state
        pos = self.robot.get_pos().cpu().numpy()
        quat = self.robot.get_quat().cpu().numpy()
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()

        # 1. Forward velocity reward (primary objective)
        # Encourage movement in positive X direction
        if hasattr(self, "_previous_pos"):
            dt = 1.0 / self.control_freq
            forward_velocity = (pos[0] - self._previous_pos[0]) / dt
            velocity_reward = min(
                forward_velocity / self.target_velocity, 2.0
            )  # Cap at 2x target
            total_reward += 1.0 * velocity_reward

        # 2. Stability reward (keep robot upright)
        # Use a more robust stability measure based on tilt from vertical
        # Smaller x,y quaternion components = more upright
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 0:
            tilt_magnitude = np.sqrt(
                quat[0] ** 2 + quat[1] ** 2
            )  # x,y components indicate tilt
            stability_reward = max(
                0.0, 1.0 - 2.0 * tilt_magnitude
            )  # Less tilt = higher reward
        else:
            stability_reward = 0.0
        total_reward += 0.5 * stability_reward

        # 3. Height maintenance reward
        # Encourage staying at reasonable walking height
        target_height = 0.8  # Approximate walking height for G1
        height_diff = abs(pos[2] - target_height)
        height_reward = max(0.0, 1.0 - height_diff)  # Linear decay
        total_reward += 0.3 * height_reward

        # 4. Energy efficiency penalty
        # Penalize excessive joint velocities (high energy consumption)
        joint_vel_penalty = np.mean(np.square(joint_vel))
        energy_penalty = min(joint_vel_penalty, 10.0)  # Cap penalty
        total_reward -= 0.1 * energy_penalty

        # 5. Action smoothness reward
        # Encourage smooth actions (penalize sudden changes)
        if self.previous_action is not None:
            action_diff = np.mean(np.square(action - self.previous_action))
            smoothness_penalty = min(action_diff, 2.0)  # Cap penalty
            total_reward -= 0.1 * smoothness_penalty

        # 6. Foot contact reward (basic version)
        # This is a simplified version - in a full implementation,
        # we'd check actual foot-ground contact forces
        # For now, penalize if robot is too high (likely jumping/falling)
        if pos[2] > 1.2:  # Too high off ground
            total_reward -= 0.5

        # Store current position for next velocity calculation
        self._previous_pos = pos.copy()

        return total_reward

    def _is_terminated(self) -> bool:
        """Check if episode should be terminated."""
        if self.robot is None:
            return True

        # Get current robot state
        pos = self.robot.get_pos().cpu().numpy()
        quat = self.robot.get_quat().cpu().numpy()

        # 1. Height termination - robot fell down
        if pos[2] < 0.3:  # Robot base too low (fallen)
            return True

        # 2. Extreme tilt termination - robot tipped over
        # Temporarily disable quaternion-based termination while we debug the quaternion format
        # TODO: Re-enable with correct quaternion interpretation
        # quat_norm = np.linalg.norm(quat)
        # if quat_norm > 0:
        #     rotation_magnitude = np.sqrt(quat[0]**2 + quat[1]**2)
        #     if rotation_magnitude > 0.8:
        #         return True

        # 3. Out of bounds termination
        if abs(pos[0]) > 10.0 or abs(pos[1]) > 5.0:  # Too far from origin
            return True

        # 4. Excessive height termination (unrealistic jumping)
        if pos[2] > 2.0:  # Too high off ground
            return True

        return False

    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.scene is not None:
            # Genesis handles rendering automatically when viewer is enabled
            pass

    def close(self):
        """Clean up resources."""
        # Genesis handles cleanup automatically
        pass
