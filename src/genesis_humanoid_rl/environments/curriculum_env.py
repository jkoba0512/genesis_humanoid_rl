"""
Curriculum-aware humanoid environment.
Adapts difficulty and rewards based on training progress.
"""

import numpy as np
import genesis as gs
from gymnasium import Env, spaces
from typing import Any, Dict, Optional, Tuple
import os
import sys

# Add robot_grounding to Python path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, project_root)
from robot_grounding import RobotGroundingCalculator

from ..curriculum.curriculum_manager import CurriculumManager, CurriculumStage
from ..rewards.curriculum_rewards import CurriculumRewardCalculator


class CurriculumHumanoidEnv(Env):
    """
    Humanoid walking environment with curriculum learning.

    Automatically adjusts difficulty based on training progress.
    """

    def __init__(
        self,
        curriculum_config_path: str = None,
        render_mode: Optional[str] = None,
        simulation_fps: int = 100,
        control_freq: int = 20,
        **kwargs,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.simulation_fps = simulation_fps
        self.control_freq = control_freq

        # Initialize curriculum manager
        self.curriculum = CurriculumManager(curriculum_config_path)

        # Initialize Genesis
        try:
            gs.init()
        except RuntimeError as e:
            if "Genesis already initialized" not in str(e):
                raise

        # Scene and robot will be initialized in reset()
        self.scene = None
        self.robot = None
        self.reward_calculator = None
        self.current_step = 0

        # Initialize spaces with default G1 robot dimensions
        self.num_joints = 35  # G1 robot has 35 DOFs

        # Define action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Define observation space: pos(3) + quat(4) + joint_pos(35) + joint_vel(35) + prev_action(35) + target_vel(1)
        obs_dim = 3 + 4 + self.num_joints + self.num_joints + self.num_joints + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.previous_action = np.zeros(self.num_joints)

        # Episode tracking
        self.episode_rewards = []
        self.current_episode_reward = 0.0

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment and check for curriculum advancement."""
        super().reset(seed=seed)

        # Check curriculum advancement (except for first episode)
        stage_advanced = False
        if hasattr(self, "current_episode_reward"):
            stage_advanced, new_stage = self.curriculum.update_episode(
                self.current_episode_reward
            )
            if stage_advanced:
                print(f"\n🎓 Advanced to stage: {new_stage.value}")
                self._print_curriculum_status()

        # Reset episode tracking
        self.current_episode_reward = 0.0

        # Get current curriculum configuration
        config = self.curriculum.get_current_config()

        # Create new scene with curriculum parameters
        scene_kwargs = {
            "sim_options": gs.options.SimOptions(
                dt=1.0 / self.simulation_fps,
                substeps=10,
            ),
            "show_viewer": False,  # Disable viewer for faster training
        }

        if self.render_mode is not None:
            scene_kwargs["viewer_options"] = gs.options.ViewerOptions(
                res=(1024, 768),
                max_FPS=self.simulation_fps,
            )
            scene_kwargs["show_viewer"] = True

        self.scene = gs.Scene(**scene_kwargs)

        # Load ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Load Unitree G1 robot
        urdf_path = os.path.join(project_root, "assets/robots/g1/g1_29dof.urdf")

        # Add some curriculum-based initial position variation
        initial_variation = self._get_initial_position_variation()

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=(
                    initial_variation[0],
                    initial_variation[1],
                    1.0 + initial_variation[2],
                ),
                euler=(0, 0, initial_variation[3]),  # Small initial rotation
            ),
        )

        # Build scene
        self.scene.build()

        # Apply automatic grounding
        if self.robot is not None:
            calculator = RobotGroundingCalculator(self.robot, verbose=False)
            grounding_height = calculator.get_grounding_height(safety_margin=0.03)
            self.robot.set_pos(
                [
                    initial_variation[0],
                    initial_variation[1],
                    grounding_height + initial_variation[2],
                ]
            )

            # Let the robot settle
            for _ in range(10):
                self.scene.step()

        # Verify robot matches expected DOF count
        if self.robot is not None:
            actual_dofs = self.robot.n_dofs
            if actual_dofs != self.num_joints:
                print(
                    f"Warning: Robot has {actual_dofs} DOFs, expected {self.num_joints}"
                )
                # Could resize spaces here if needed

        # Initialize reward calculator
        self.reward_calculator = CurriculumRewardCalculator(self.curriculum)

        # Reset episode state
        self.current_step = 0
        if self.previous_action is None:
            self.previous_action = np.zeros(self.num_joints)

        # Initialize previous position for velocity calculation
        if self.robot is not None:
            self._previous_pos = self.robot.get_pos().cpu().numpy()

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with curriculum-aware dynamics."""
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Apply action to robot joints with curriculum-specific scaling
        if self.robot is not None:
            action_scaling = self._get_action_scaling()
            current_pos = self.robot.get_dofs_position().cpu()  # Move to CPU first
            target_pos = current_pos + (action * action_scaling)
            self.robot.control_dofs_position(target_pos)

        # Step simulation
        steps_per_control = self.simulation_fps // self.control_freq
        for _ in range(steps_per_control):
            self.scene.step()

        # Get observation
        obs = self._get_observation()

        # Calculate curriculum-aware reward
        info = {"previous_action": self.previous_action}
        reward = self.reward_calculator.calculate_reward(self.robot, action, info)
        self.current_episode_reward += reward

        # Update step counter
        self.current_step += 1

        # Check curriculum-aware termination
        terminated = self.reward_calculator.is_terminated(self.robot)

        # Check episode length (curriculum-dependent)
        config = self.curriculum.get_current_config()
        truncated = self.current_step >= config.max_episode_steps

        # Store previous action
        self.previous_action = action.copy()

        # Comprehensive info
        curriculum_info = self.curriculum.get_progress_info()
        info.update(
            {
                "step": self.current_step,
                "terminated": terminated,
                "truncated": truncated,
                "episode_reward": self.current_episode_reward,
                "curriculum": curriculum_info,
            }
        )

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get curriculum-aware observation."""
        if self.robot is None:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get robot state
        pos = self.robot.get_pos().cpu().numpy()
        quat = self.robot.get_quat().cpu().numpy()
        joint_pos = self.robot.get_dofs_position().cpu().numpy()
        joint_vel = self.robot.get_dofs_velocity().cpu().numpy()

        # Get curriculum-aware target velocity
        target_velocity = self.curriculum.get_adaptive_target_velocity()

        # Concatenate observations
        obs = np.concatenate(
            [
                pos,  # Base position (3)
                quat,  # Base orientation (4)
                joint_pos,  # Joint positions
                joint_vel,  # Joint velocities
                self.previous_action,  # Previous action
                [target_velocity],  # Target velocity (curriculum-dependent)
            ]
        )

        return obs.astype(np.float32)

    def _get_initial_position_variation(self) -> np.ndarray:
        """Get curriculum-based initial position variation."""
        stage = self.curriculum.current_stage

        if stage == CurriculumStage.BALANCE:
            # Minimal variation for balance training
            return np.array([0.0, 0.0, 0.0, 0.0])
        elif stage == CurriculumStage.SMALL_STEPS:
            # Small position variations
            return np.array(
                [
                    np.random.uniform(-0.05, 0.05),  # x
                    np.random.uniform(-0.05, 0.05),  # y
                    np.random.uniform(-0.02, 0.02),  # z
                    np.random.uniform(-0.1, 0.1),  # rotation
                ]
            )
        else:
            # Larger variations for advanced stages
            return np.array(
                [
                    np.random.uniform(-0.1, 0.1),  # x
                    np.random.uniform(-0.1, 0.1),  # y
                    np.random.uniform(-0.05, 0.05),  # z
                    np.random.uniform(-0.2, 0.2),  # rotation
                ]
            )

    def _get_action_scaling(self) -> float:
        """Get curriculum-based action scaling."""
        stage = self.curriculum.current_stage

        if stage == CurriculumStage.BALANCE:
            return 0.05  # Very small movements
        elif stage == CurriculumStage.SMALL_STEPS:
            return 0.08  # Small movements
        elif stage == CurriculumStage.WALKING:
            return 0.1  # Normal movements
        else:
            return 0.12  # Larger movements for advanced behaviors

    def _print_curriculum_status(self):
        """Print current curriculum status."""
        info = self.curriculum.get_progress_info()
        print(f"📊 Curriculum Status:")
        print(f"   Stage: {info['current_stage']}")
        print(f"   Progress: {info['stage_progress']}")
        print(f"   Recent Avg Reward: {info['recent_avg_reward']:.3f}")
        print(f"   Target Velocity: {info['target_velocity']:.2f}")
        print(f"   Total Episodes: {info['total_episodes']}")

    def get_curriculum_info(self) -> Dict[str, Any]:
        """Get current curriculum information."""
        return self.curriculum.get_progress_info()

    def save_curriculum_progress(self, path: str):
        """Save curriculum progress to file."""
        self.curriculum.save_config(path)

    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and self.scene is not None:
            pass  # Genesis handles rendering automatically

    def close(self):
        """Clean up resources."""
        if hasattr(self, "reward_calculator") and self.reward_calculator:
            self.reward_calculator.reset_episode()
        # Genesis handles cleanup automatically
