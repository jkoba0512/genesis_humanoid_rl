"""
Curriculum Learning Manager for Humanoid Walking Training.
Gradually increases task difficulty as the robot improves.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class CurriculumStage(Enum):
    """Curriculum learning stages in order of difficulty."""

    BALANCE = "balance"  # Stage 1: Learn to stand upright
    SMALL_STEPS = "small_steps"  # Stage 2: Tiny forward movements
    WALKING = "walking"  # Stage 3: Continuous walking
    SPEED_CONTROL = "speed_control"  # Stage 4: Variable walking speeds
    TURNING = "turning"  # Stage 5: Directional control
    OBSTACLES = "obstacles"  # Stage 6: Navigate around objects
    TERRAIN = "terrain"  # Stage 7: Different ground types


@dataclass
class StageConfig:
    """Configuration for each curriculum stage."""

    name: str
    target_velocity: float = 0.0
    max_episode_steps: int = 200
    success_threshold: float = 0.8  # Avg reward needed to advance
    min_episodes: int = 50  # Minimum episodes before advancement
    reward_weights: Dict[str, float] = field(default_factory=dict)
    termination_conditions: Dict[str, Any] = field(default_factory=dict)
    disturbances: Dict[str, float] = field(default_factory=dict)


class CurriculumManager:
    """Manages curriculum learning progression."""

    def __init__(self, config_path: str = None):
        self.current_stage = CurriculumStage.BALANCE
        self.stage_configs = self._create_default_stages()
        self.episode_count = 0
        self.stage_episode_count = 0
        self.recent_rewards = []
        self.stage_history = []

        if config_path:
            self.load_config(config_path)

    def _create_default_stages(self) -> Dict[CurriculumStage, StageConfig]:
        """Create default curriculum stages."""
        stages = {}

        # Stage 1: Balance - Learn to stand upright
        stages[CurriculumStage.BALANCE] = StageConfig(
            name="Balance Training",
            target_velocity=0.0,
            max_episode_steps=100,
            success_threshold=0.5,
            min_episodes=20,
            reward_weights={
                "stability": 2.0,  # Heavy emphasis on staying upright
                "height": 1.0,  # Maintain proper height
                "energy": -0.05,  # Light energy penalty
                "velocity": 0.0,  # No velocity reward yet
                "smoothness": -0.05,  # Encourage smooth movements
            },
            termination_conditions={
                "min_height": 0.4,  # More lenient height threshold
                "max_tilt": 0.6,  # Allow more tilt initially
            },
        )

        # Stage 2: Small Steps - Tiny forward movements
        stages[CurriculumStage.SMALL_STEPS] = StageConfig(
            name="Small Steps",
            target_velocity=0.3,
            max_episode_steps=150,
            success_threshold=0.6,
            min_episodes=30,
            reward_weights={
                "stability": 1.5,
                "height": 0.8,
                "energy": -0.08,
                "velocity": 0.5,  # Small velocity reward
                "smoothness": -0.1,
            },
            termination_conditions={
                "min_height": 0.5,
                "max_tilt": 0.5,
            },
        )

        # Stage 3: Walking - Continuous forward motion
        stages[CurriculumStage.WALKING] = StageConfig(
            name="Basic Walking",
            target_velocity=1.0,
            max_episode_steps=200,
            success_threshold=0.7,
            min_episodes=50,
            reward_weights={
                "stability": 1.0,
                "height": 0.6,
                "energy": -0.1,
                "velocity": 1.0,  # Full velocity reward
                "smoothness": -0.1,
            },
            termination_conditions={
                "min_height": 0.6,
                "max_tilt": 0.4,
            },
        )

        # Stage 4: Speed Control - Variable walking speeds
        stages[CurriculumStage.SPEED_CONTROL] = StageConfig(
            name="Speed Control",
            target_velocity=1.5,  # Will vary during training
            max_episode_steps=250,
            success_threshold=0.75,
            min_episodes=40,
            reward_weights={
                "stability": 0.8,
                "height": 0.5,
                "energy": -0.15,  # More energy efficiency focus
                "velocity": 1.2,  # Higher velocity reward
                "smoothness": -0.15,
            },
            termination_conditions={
                "min_height": 0.65,
                "max_tilt": 0.35,
            },
        )

        # Stage 5: Turning - Directional control
        stages[CurriculumStage.TURNING] = StageConfig(
            name="Turning Control",
            target_velocity=1.0,
            max_episode_steps=300,
            success_threshold=0.8,
            min_episodes=60,
            reward_weights={
                "stability": 0.8,
                "height": 0.4,
                "energy": -0.2,
                "velocity": 0.8,
                "smoothness": -0.2,
                "direction": 0.6,  # New: directional control reward
            },
            termination_conditions={
                "min_height": 0.7,
                "max_tilt": 0.3,
            },
        )

        return stages

    def get_current_config(self) -> StageConfig:
        """Get configuration for current stage."""
        return self.stage_configs[self.current_stage]

    def update_episode(self, episode_reward: float) -> Tuple[bool, CurriculumStage]:
        """
        Update curriculum based on episode performance.

        Returns:
            (stage_advanced, new_stage)
        """
        self.episode_count += 1
        self.stage_episode_count += 1
        self.recent_rewards.append(episode_reward)

        # Keep only recent rewards for evaluation
        max_recent = 50
        if len(self.recent_rewards) > max_recent:
            self.recent_rewards = self.recent_rewards[-max_recent:]

        # Check if ready to advance stage
        if self._should_advance_stage():
            old_stage = self.current_stage
            self._advance_stage()
            print(
                f"\n🎓 CURRICULUM ADVANCE: {old_stage.value} → {self.current_stage.value}"
            )
            print(f"   Episodes in {old_stage.value}: {self.stage_episode_count}")
            print(f"   Average reward: {np.mean(self.recent_rewards):.3f}")
            self.stage_history.append(
                {
                    "stage": old_stage.value,
                    "episodes": self.stage_episode_count,
                    "avg_reward": np.mean(self.recent_rewards),
                }
            )
            self.stage_episode_count = 0
            self.recent_rewards = []
            return True, self.current_stage

        return False, self.current_stage

    def _should_advance_stage(self) -> bool:
        """Check if current stage should be advanced."""
        config = self.get_current_config()

        # Need minimum episodes
        if self.stage_episode_count < config.min_episodes:
            return False

        # Need minimum number of recent rewards
        if len(self.recent_rewards) < min(20, config.min_episodes):
            return False

        # Check if average reward meets threshold
        avg_reward = np.mean(self.recent_rewards)
        return avg_reward >= config.success_threshold

    def _advance_stage(self):
        """Advance to next curriculum stage."""
        stages = list(CurriculumStage)
        current_idx = stages.index(self.current_stage)

        if current_idx < len(stages) - 1:
            self.current_stage = stages[current_idx + 1]
        # If at final stage, stay there

    def get_adaptive_target_velocity(self) -> float:
        """Get target velocity that adapts based on current stage."""
        config = self.get_current_config()
        base_velocity = config.target_velocity

        # For speed control stage, vary the target velocity
        if self.current_stage == CurriculumStage.SPEED_CONTROL:
            # Cycle through different speeds
            cycle = (self.episode_count // 10) % 4
            velocities = [0.5, 1.0, 1.5, 2.0]
            return velocities[cycle]

        return base_velocity

    def get_progress_info(self) -> Dict[str, Any]:
        """Get current curriculum progress information."""
        config = self.get_current_config()

        return {
            "current_stage": self.current_stage.value,
            "stage_progress": f"{self.stage_episode_count}/{config.min_episodes}",
            "total_episodes": self.episode_count,
            "recent_avg_reward": (
                np.mean(self.recent_rewards) if self.recent_rewards else 0.0
            ),
            "success_threshold": config.success_threshold,
            "target_velocity": self.get_adaptive_target_velocity(),
            "stage_history": self.stage_history,
        }

    def save_config(self, path: str):
        """Save curriculum configuration to file."""
        config_data = {
            "current_stage": self.current_stage.value,
            "episode_count": self.episode_count,
            "stage_episode_count": self.stage_episode_count,
            "recent_rewards": self.recent_rewards,
            "stage_history": self.stage_history,
        }

        with open(path, "w") as f:
            json.dump(config_data, f, indent=2)

    def load_config(self, path: str):
        """Load curriculum configuration from file."""
        try:
            with open(path, "r") as f:
                config_data = json.load(f)

            self.current_stage = CurriculumStage(config_data["current_stage"])
            self.episode_count = config_data["episode_count"]
            self.stage_episode_count = config_data["stage_episode_count"]
            self.recent_rewards = config_data["recent_rewards"]
            self.stage_history = config_data.get("stage_history", [])

        except FileNotFoundError:
            print(f"Curriculum config file {path} not found, using defaults")
        except Exception as e:
            print(f"Error loading curriculum config: {e}, using defaults")

    def reset_stage(self, stage: CurriculumStage = None):
        """Reset to specified stage or beginning."""
        if stage is None:
            stage = CurriculumStage.BALANCE

        self.current_stage = stage
        self.stage_episode_count = 0
        self.recent_rewards = []
        print(f"🔄 Curriculum reset to stage: {stage.value}")
