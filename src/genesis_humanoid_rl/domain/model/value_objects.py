"""
Domain value objects for humanoid robotics learning.
Immutable objects that define domain concepts with rich behavior.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# Identity value objects
@dataclass(frozen=True)
class SessionId:
    """Unique identifier for learning sessions."""

    value: str

    @classmethod
    def generate(cls) -> "SessionId":
        """Generate a new unique session ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "SessionId":
        """Create SessionId from string."""
        return cls(value)


@dataclass(frozen=True)
class RobotId:
    """Unique identifier for humanoid robots."""

    value: str

    @classmethod
    def generate(cls) -> "RobotId":
        """Generate a new unique robot ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "RobotId":
        """Create RobotId from string."""
        return cls(value)


@dataclass(frozen=True)
class PlanId:
    """Unique identifier for curriculum plans."""

    value: str

    @classmethod
    def generate(cls) -> "PlanId":
        """Generate a new unique plan ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "PlanId":
        """Create PlanId from string."""
        return cls(value)


@dataclass(frozen=True)
class EpisodeId:
    """Unique identifier for learning episodes."""

    value: str

    @classmethod
    def generate(cls) -> "EpisodeId":
        """Generate a new unique episode ID."""
        return cls(str(uuid.uuid4()))

    @classmethod
    def from_string(cls, value: str) -> "EpisodeId":
        """Create EpisodeId from string."""
        return cls(value)


# Motion and behavior value objects
class MotionType(Enum):
    """Types of motion commands."""

    BALANCE = "balance"
    WALK_FORWARD = "walk_forward"
    WALK_BACKWARD = "walk_backward"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STOP = "stop"
    CUSTOM = "custom"


@dataclass(frozen=True)
class MotionCommand:
    """
    High-level motion command with semantic meaning.
    Encapsulates intent rather than low-level control actions.
    """

    motion_type: MotionType
    velocity: float = 1.0  # Target velocity in m/s
    duration: Optional[float] = None  # Duration in seconds
    parameters: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Validate motion command parameters."""
        if self.velocity < 0:
            raise ValueError("Velocity must be non-negative")

        if self.duration is not None and self.duration <= 0:
            raise ValueError("Duration must be positive")

    def is_locomotion_command(self) -> bool:
        """Check if this is a locomotion (movement) command."""
        return self.motion_type in [
            MotionType.WALK_FORWARD,
            MotionType.WALK_BACKWARD,
            MotionType.TURN_LEFT,
            MotionType.TURN_RIGHT,
        ]

    def requires_balance(self) -> bool:
        """Check if this command requires balance capability."""
        return self.motion_type != MotionType.STOP

    def get_complexity_score(self) -> float:
        """Get relative complexity of this motion command."""
        complexity_map = {
            MotionType.BALANCE: 1.0,
            MotionType.STOP: 0.5,
            MotionType.WALK_FORWARD: 2.0,
            MotionType.WALK_BACKWARD: 2.5,
            MotionType.TURN_LEFT: 3.0,
            MotionType.TURN_RIGHT: 3.0,
            MotionType.CUSTOM: 4.0,
        }

        base_complexity = complexity_map.get(self.motion_type, 2.0)
        velocity_factor = min(self.velocity / 2.0, 2.0)  # Cap at 2x

        return base_complexity * velocity_factor


class SkillType(Enum):
    """Types of locomotion skills."""

    POSTURAL_CONTROL = "postural_control"
    STATIC_BALANCE = "static_balance"
    DYNAMIC_BALANCE = "dynamic_balance"
    FORWARD_WALKING = "forward_walking"
    BACKWARD_WALKING = "backward_walking"
    TURNING = "turning"
    SPEED_CONTROL = "speed_control"
    TERRAIN_ADAPTATION = "terrain_adaptation"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"


class MasteryLevel(Enum):
    """Levels of skill mastery."""

    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

    def get_numeric_value(self) -> float:
        """Get numeric representation of mastery level."""
        mapping = {
            MasteryLevel.NOVICE: 0.0,
            MasteryLevel.BEGINNER: 0.25,
            MasteryLevel.INTERMEDIATE: 0.5,
            MasteryLevel.ADVANCED: 0.75,
            MasteryLevel.EXPERT: 1.0,
        }
        return mapping[self]

    @classmethod
    def from_score(cls, score: float) -> "MasteryLevel":
        """Convert numeric score to mastery level."""
        if score < 0.125:
            return cls.NOVICE
        elif score < 0.375:
            return cls.BEGINNER
        elif score < 0.625:
            return cls.INTERMEDIATE
        elif score < 0.875:
            return cls.ADVANCED
        else:
            return cls.EXPERT


@dataclass(frozen=True)
class LocomotionSkill:
    """
    Domain concept representing a learnable locomotion skill.
    """

    skill_type: SkillType
    mastery_level: MasteryLevel = MasteryLevel.NOVICE
    proficiency_score: float = 0.0
    last_assessed: Optional[datetime] = None

    def __post_init__(self):
        """Validate skill parameters."""
        if not 0.0 <= self.proficiency_score <= 1.0:
            raise ValueError("Proficiency score must be between 0.0 and 1.0")

    def is_mastered(self, threshold: MasteryLevel = MasteryLevel.INTERMEDIATE) -> bool:
        """Check if skill is mastered at specified threshold."""
        return self.mastery_level.get_numeric_value() >= threshold.get_numeric_value()

    def requires_skill(self, other_skill: SkillType) -> bool:
        """Check if this skill requires another skill as prerequisite."""
        prerequisites = {
            SkillType.STATIC_BALANCE: [SkillType.POSTURAL_CONTROL],
            SkillType.DYNAMIC_BALANCE: [
                SkillType.STATIC_BALANCE,
                SkillType.POSTURAL_CONTROL,
            ],
            SkillType.FORWARD_WALKING: [SkillType.DYNAMIC_BALANCE],
            SkillType.BACKWARD_WALKING: [SkillType.FORWARD_WALKING],
            SkillType.TURNING: [SkillType.FORWARD_WALKING],
            SkillType.SPEED_CONTROL: [SkillType.FORWARD_WALKING],
            SkillType.TERRAIN_ADAPTATION: [SkillType.SPEED_CONTROL],
            SkillType.OBSTACLE_AVOIDANCE: [SkillType.TURNING, SkillType.SPEED_CONTROL],
        }

        return other_skill in prerequisites.get(self.skill_type, [])


@dataclass(frozen=True)
class GaitPattern:
    """
    Domain concept representing a specific walking pattern.
    """

    stride_length: float  # meters
    stride_frequency: float  # steps per second
    step_height: float  # meters
    stability_margin: float  # meters (minimum distance to instability)
    energy_efficiency: float  # 0.0 to 1.0
    symmetry_score: float  # 0.0 to 1.0

    def __post_init__(self):
        """Validate gait parameters."""
        if self.stride_length <= 0:
            raise ValueError("Stride length must be positive")
        if self.stride_frequency <= 0:
            raise ValueError("Stride frequency must be positive")
        if self.step_height < 0:
            raise ValueError("Step height must be non-negative")
        if not 0.0 <= self.energy_efficiency <= 1.0:
            raise ValueError("Energy efficiency must be between 0.0 and 1.0")
        if not 0.0 <= self.symmetry_score <= 1.0:
            raise ValueError("Symmetry score must be between 0.0 and 1.0")

    def get_walking_speed(self) -> float:
        """Calculate walking speed from gait parameters."""
        return self.stride_length * self.stride_frequency

    def is_stable_gait(self, min_stability: float = 0.05) -> bool:
        """Check if gait pattern is stable."""
        return self.stability_margin >= min_stability

    def get_quality_score(self) -> float:
        """Calculate overall gait quality score."""
        stability_score = min(self.stability_margin / 0.1, 1.0)  # Normalize to 0.1m

        return (
            stability_score * 0.4
            + self.energy_efficiency * 0.3
            + self.symmetry_score * 0.3
        )


@dataclass(frozen=True)
class MovementTrajectory:
    """
    Domain concept representing a sequence of robot movements.
    """

    positions: List[Tuple[float, float, float]]  # (x, y, z) coordinates
    timestamps: List[float]  # Time points in seconds
    velocities: Optional[List[Tuple[float, float, float]]] = None

    def __post_init__(self):
        """Validate trajectory data."""
        if len(self.positions) != len(self.timestamps):
            raise ValueError("Positions and timestamps must have same length")

        if len(self.positions) < 2:
            raise ValueError("Trajectory must have at least 2 points")

        if self.velocities and len(self.velocities) != len(self.positions):
            raise ValueError("Velocities must match positions length")

    def get_total_distance(self) -> float:
        """Calculate total distance traveled."""
        total_distance = 0.0

        for i in range(1, len(self.positions)):
            prev_pos = np.array(self.positions[i - 1])
            curr_pos = np.array(self.positions[i])
            distance = np.linalg.norm(curr_pos - prev_pos)
            total_distance += distance

        return total_distance

    def get_average_velocity(self) -> float:
        """Calculate average velocity over trajectory."""
        total_distance = self.get_total_distance()
        total_time = self.timestamps[-1] - self.timestamps[0]

        if total_time <= 0:
            return 0.0

        return total_distance / total_time

    def get_smoothness_score(self) -> float:
        """Calculate trajectory smoothness (lower is smoother)."""
        if len(self.positions) < 3:
            return 0.0

        accelerations = []

        for i in range(1, len(self.positions) - 1):
            # Calculate acceleration at each point
            dt1 = self.timestamps[i] - self.timestamps[i - 1]
            dt2 = self.timestamps[i + 1] - self.timestamps[i]

            if dt1 <= 0 or dt2 <= 0:
                continue

            pos_prev = np.array(self.positions[i - 1])
            pos_curr = np.array(self.positions[i])
            pos_next = np.array(self.positions[i + 1])

            vel1 = (pos_curr - pos_prev) / dt1
            vel2 = (pos_next - pos_curr) / dt2

            accel = (vel2 - vel1) / ((dt1 + dt2) / 2)
            accelerations.append(np.linalg.norm(accel))

        if not accelerations:
            return 0.0

        # Return inverse of average acceleration (higher score = smoother)
        avg_acceleration = np.mean(accelerations)
        return 1.0 / (1.0 + avg_acceleration)


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Domain concept for comprehensive performance assessment.
    """

    success_rate: float  # 0.0 to 1.0
    average_reward: float
    skill_scores: Dict[SkillType, float] = field(default_factory=dict)
    gait_quality: Optional[float] = None
    learning_progress: float = 0.0  # Rate of improvement
    stability_incidents: int = 0  # Number of falls or instabilities

    def __post_init__(self):
        """Validate performance metrics."""
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("Success rate must be between 0.0 and 1.0")

        for skill, score in self.skill_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Skill score for {skill} must be between 0.0 and 1.0")

    def get_overall_performance(self) -> float:
        """Calculate overall performance score."""
        # Base components with fixed weights that sum to 0.7
        # Normalize reward more sensibly - assume rewards > 10.0 indicate very high performance
        normalized_reward = min(max(self.average_reward, 0.0) / 10.0, 1.0)

        base_score = (
            self.success_rate * 0.3
            + normalized_reward * 0.2
            + self.learning_progress * 0.2
        )

        # Optional components
        skill_score = 0.0
        if self.skill_scores:
            avg_skill_score = np.mean(list(self.skill_scores.values()))
            skill_score = avg_skill_score * 0.2

        gait_score = 0.0
        if self.gait_quality is not None:
            gait_score = self.gait_quality * 0.1

        return base_score + skill_score + gait_score

    def is_improving(self, threshold: float = 0.1) -> bool:
        """Check if performance shows improvement."""
        return self.learning_progress >= threshold

    def get_dominant_skill(self) -> Optional[SkillType]:
        """Get the skill with highest proficiency."""
        if not self.skill_scores:
            return None

        return max(self.skill_scores.items(), key=lambda x: x[1])[0]


@dataclass(frozen=True)
class SkillAssessment:
    """
    Domain concept for evaluating skill mastery.
    """

    skill: LocomotionSkill
    assessment_score: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    evidence_quality: float  # 0.0 to 1.0
    assessment_date: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate assessment parameters."""
        if not 0.0 <= self.assessment_score <= 1.0:
            raise ValueError("Assessment score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")
        if not 0.0 <= self.evidence_quality <= 1.0:
            raise ValueError("Evidence quality must be between 0.0 and 1.0")

    def is_reliable_assessment(
        self, min_confidence: float = 0.7, min_evidence: float = 0.6
    ) -> bool:
        """Check if assessment is reliable."""
        return (
            self.confidence_level >= min_confidence
            and self.evidence_quality >= min_evidence
        )

    def get_adjusted_score(self) -> float:
        """Get assessment score adjusted for confidence and evidence quality."""
        reliability_factor = (self.confidence_level + self.evidence_quality) / 2.0
        return self.assessment_score * reliability_factor

    def suggests_mastery(self, threshold: float = 0.7) -> bool:
        """Check if assessment suggests skill mastery."""
        # For very low thresholds, use less aggressive adjustment
        if threshold <= 0.5:
            # Use a lighter adjustment for low thresholds
            reliability_factor = (
                self.confidence_level + self.evidence_quality + 1.0
            ) / 3.0
            adjusted_score = self.assessment_score * reliability_factor
            min_confidence = 0.5
            min_evidence = 0.4
        else:
            # Use standard adjustment for normal thresholds
            adjusted_score = self.get_adjusted_score()
            min_confidence = 0.7
            min_evidence = 0.6

        is_reliable = self.is_reliable_assessment(min_confidence, min_evidence)

        return adjusted_score >= threshold and is_reliable


@dataclass(frozen=True)
class DifficultyParameters:
    """Parameters for controlling training difficulty."""

    base_difficulty: float = 1.0
    progression_rate: float = 0.1
    adaptation_threshold: float = 0.8
    max_difficulty: float = 5.0
    min_difficulty: float = 0.1

    def __post_init__(self):
        """Validate difficulty parameters."""
        if not (0.0 <= self.base_difficulty <= 10.0):
            raise ValueError("base_difficulty must be between 0.0 and 10.0")
        if not (0.0 <= self.progression_rate <= 1.0):
            raise ValueError("progression_rate must be between 0.0 and 1.0")
        if self.min_difficulty >= self.max_difficulty:
            raise ValueError("min_difficulty must be less than max_difficulty")

    def calculate_adapted_difficulty(self, performance_score: float) -> float:
        """Calculate adapted difficulty based on performance."""
        if performance_score > self.adaptation_threshold:
            # Increase difficulty
            new_difficulty = self.base_difficulty * (1 + self.progression_rate)
        else:
            # Decrease difficulty
            new_difficulty = self.base_difficulty * (1 - self.progression_rate)

        # Clamp to valid range
        return max(self.min_difficulty, min(self.max_difficulty, new_difficulty))

    def get_complexity_multiplier(self) -> float:
        """Get complexity multiplier based on difficulty."""
        return 0.5 + (self.base_difficulty / self.max_difficulty) * 1.5


@dataclass(frozen=True)
class RobotConfiguration:
    """Configuration parameters for humanoid robots."""

    joint_count: int
    height: float
    weight: float
    max_joint_velocity: float = 10.0
    max_joint_torque: float = 100.0
    control_frequency: int = 20
    simulation_frequency: int = 100

    def __post_init__(self):
        """Validate robot configuration."""
        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive")
        if self.height <= 0:
            raise ValueError("height must be positive")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        if self.max_joint_velocity <= 0:
            raise ValueError("max_joint_velocity must be positive")
        if self.max_joint_torque <= 0:
            raise ValueError("max_joint_torque must be positive")
        if self.control_frequency <= 0:
            raise ValueError("control_frequency must be positive")
        if self.simulation_frequency <= 0:
            raise ValueError("simulation_frequency must be positive")

    def get_control_timestep(self) -> float:
        """Get control timestep in seconds."""
        return 1.0 / self.control_frequency

    def get_simulation_timestep(self) -> float:
        """Get simulation timestep in seconds."""
        return 1.0 / self.simulation_frequency

    def get_action_space_size(self) -> int:
        """Get action space size for RL."""
        return self.joint_count

    def validate_action(self, action: np.ndarray) -> bool:
        """Validate if action is within robot limits."""
        if len(action) != self.joint_count:
            return False

        # Check if action values are within velocity limits
        return np.all(np.abs(action) <= self.max_joint_velocity)
