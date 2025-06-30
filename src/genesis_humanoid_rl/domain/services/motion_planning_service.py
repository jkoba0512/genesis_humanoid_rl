"""
Motion Planning Domain Service.
Encapsulates business rules for converting skills to motion commands.
"""

from typing import Dict, Optional
from dataclasses import dataclass

from ..model.value_objects import SkillType, MotionCommand, MotionType


@dataclass
class TrainingContext:
    """Context information for motion planning."""

    max_steps: int = 100
    difficulty_level: float = 1.0
    environment_constraints: Optional[Dict[str, float]] = None


class MotionPlanningService:
    """
    Domain service for planning motion commands based on target skills.

    Encapsulates the business logic for mapping skills to appropriate
    motion commands with proper parameters.
    """

    # Skill-to-motion mapping (business rules)
    SKILL_MOTION_MAPPING = {
        SkillType.STATIC_BALANCE: {
            "motion_type": MotionType.BALANCE,
            "base_velocity": 0.0,
            "complexity_factor": 0.1,
        },
        SkillType.DYNAMIC_BALANCE: {
            "motion_type": MotionType.BALANCE,
            "base_velocity": 0.2,
            "complexity_factor": 0.3,
        },
        SkillType.FORWARD_WALKING: {
            "motion_type": MotionType.WALK_FORWARD,
            "base_velocity": 1.0,
            "complexity_factor": 0.5,
        },
        SkillType.BACKWARD_WALKING: {
            "motion_type": MotionType.WALK_BACKWARD,
            "base_velocity": 0.8,
            "complexity_factor": 0.6,
        },
        SkillType.SPEED_CONTROL: {
            "motion_type": MotionType.WALK_FORWARD,
            "base_velocity": 1.5,
            "complexity_factor": 0.7,
        },
        SkillType.TURNING: {
            "motion_type": MotionType.TURN_LEFT,  # Default to left turn, could be randomized
            "base_velocity": 0.5,
            "complexity_factor": 0.8,
        },
        SkillType.OBSTACLE_AVOIDANCE: {
            "motion_type": MotionType.WALK_FORWARD,
            "base_velocity": 0.7,
            "complexity_factor": 0.9,
        },
    }

    def create_motion_command(
        self, target_skill: SkillType, context: Optional[TrainingContext] = None
    ) -> MotionCommand:
        """
        Create a motion command for the target skill.

        Args:
            target_skill: The skill to target during motion
            context: Training context for parameter adjustment

        Returns:
            MotionCommand configured for the target skill

        Raises:
            ValueError: If skill is not supported
        """
        if context is None:
            context = TrainingContext()

        # Get skill configuration
        skill_config = self.SKILL_MOTION_MAPPING.get(target_skill)
        if not skill_config:
            # Fallback to forward walking for unknown skills
            skill_config = self.SKILL_MOTION_MAPPING[SkillType.FORWARD_WALKING]

        # Calculate adjusted parameters based on context
        adjusted_velocity = self._calculate_velocity(skill_config, context)
        duration = self._calculate_duration(context)
        parameters = self._build_parameters(skill_config, context)

        return MotionCommand(
            motion_type=skill_config["motion_type"],
            velocity=adjusted_velocity,
            duration=duration,
            parameters=parameters,
        )

    def _calculate_velocity(
        self, skill_config: Dict, context: TrainingContext
    ) -> float:
        """Calculate adjusted velocity based on difficulty and constraints."""
        base_velocity = skill_config["base_velocity"]
        complexity_factor = skill_config["complexity_factor"]

        # Adjust for difficulty level
        difficulty_adjustment = context.difficulty_level * complexity_factor
        adjusted_velocity = base_velocity * (1.0 + difficulty_adjustment * 0.5)

        # Apply environment constraints if present
        if (
            context.environment_constraints
            and "max_velocity" in context.environment_constraints
        ):
            max_velocity = context.environment_constraints["max_velocity"]
            adjusted_velocity = min(adjusted_velocity, max_velocity)

        return max(0.0, adjusted_velocity)  # Ensure non-negative

    def _calculate_duration(self, context: TrainingContext) -> float:
        """Calculate motion duration based on context."""
        # Convert steps to seconds assuming 50Hz control frequency
        return context.max_steps / 50.0

    def _build_parameters(self, skill_config: Dict, context: TrainingContext) -> Dict:
        """Build additional parameters for the motion command."""
        parameters = {
            "difficulty_level": context.difficulty_level,
            "complexity_factor": skill_config["complexity_factor"],
        }

        # Add environment constraints if present
        if context.environment_constraints:
            parameters["environment_constraints"] = context.environment_constraints

        return parameters

    def get_supported_skills(self) -> list[SkillType]:
        """Get list of skills supported by this service."""
        return list(self.SKILL_MOTION_MAPPING.keys())

    def is_skill_supported(self, skill: SkillType) -> bool:
        """Check if a skill is supported by this service."""
        return skill in self.SKILL_MOTION_MAPPING
