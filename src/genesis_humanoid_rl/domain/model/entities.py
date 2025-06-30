"""
Domain entities for humanoid robotics learning.
Entities have identity and lifecycle, containing business logic and state changes.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from enum import Enum
import logging

from .value_objects import (
    EpisodeId,
    SessionId,
    RobotId,
    MotionCommand,
    LocomotionSkill,
    SkillType,
    MasteryLevel,
    PerformanceMetrics,
    GaitPattern,
    MovementTrajectory,
    SkillAssessment,
)

logger = logging.getLogger(__name__)


class EpisodeStatus(Enum):
    """Status of a learning episode."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


class EpisodeOutcome(Enum):
    """Outcome of a learning episode."""

    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    TERMINATED_EARLY = "terminated_early"
    ERROR = "error"


@dataclass
class LearningEpisode:
    """
    Entity representing a single learning episode.

    An episode is a discrete learning session with clear start/end boundaries,
    containing the robot's actions, observations, and performance outcomes.
    """

    episode_id: EpisodeId
    session_id: SessionId
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    status: EpisodeStatus = EpisodeStatus.PENDING
    outcome: Optional[EpisodeOutcome] = None

    # Episode configuration
    target_skill: Optional[SkillType] = None
    motion_commands: List[MotionCommand] = field(default_factory=list)
    max_steps: int = 1000

    # Performance tracking
    total_reward: float = 0.0
    step_count: int = 0
    performance_metrics: Optional[PerformanceMetrics] = None
    movement_trajectory: Optional[MovementTrajectory] = None

    # Episode data
    episode_data: Dict[str, Any] = field(default_factory=dict)

    def start_episode(self, target_skill: Optional[SkillType] = None) -> None:
        """Start the learning episode."""
        if self.status != EpisodeStatus.PENDING:
            raise ValueError(f"Cannot start episode in status {self.status}")

        self.status = EpisodeStatus.RUNNING
        self.start_time = datetime.now()
        self.target_skill = target_skill
        self.step_count = 0
        self.total_reward = 0.0

        logger.info(
            f"Started episode {self.episode_id.value} targeting skill {target_skill}"
        )

    def add_step_reward(self, reward: float) -> None:
        """Add reward for a single step."""
        if self.status != EpisodeStatus.RUNNING:
            raise ValueError(f"Cannot add reward to episode in status {self.status}")

        self.total_reward += reward
        self.step_count += 1

    def execute_motion_command(self, command: MotionCommand) -> None:
        """Record execution of a motion command."""
        if self.status != EpisodeStatus.RUNNING:
            raise ValueError(f"Cannot execute command in episode status {self.status}")

        self.motion_commands.append(command)

        # Store command in episode data for analysis
        if "commands" not in self.episode_data:
            self.episode_data["commands"] = []
        self.episode_data["commands"].append(
            {"command": command, "step": self.step_count, "timestamp": datetime.now()}
        )

    def complete_episode(
        self,
        outcome: EpisodeOutcome,
        performance_metrics: Optional[PerformanceMetrics] = None,
    ) -> None:
        """Complete the learning episode."""
        if self.status != EpisodeStatus.RUNNING:
            raise ValueError(f"Cannot complete episode in status {self.status}")

        self.status = EpisodeStatus.COMPLETED
        self.end_time = datetime.now()
        self.outcome = outcome
        self.performance_metrics = performance_metrics

        logger.info(f"Completed episode {self.episode_id.value} with outcome {outcome}")

    def terminate_episode(self, reason: str) -> None:
        """Terminate the episode early."""
        if self.status not in [EpisodeStatus.RUNNING, EpisodeStatus.PENDING]:
            return  # Already terminated or completed

        self.status = EpisodeStatus.TERMINATED
        self.end_time = datetime.now()
        self.outcome = EpisodeOutcome.TERMINATED_EARLY
        self.episode_data["termination_reason"] = reason

        logger.warning(f"Terminated episode {self.episode_id.value}: {reason}")

    def fail_episode(self, error_message: str) -> None:
        """Mark episode as failed due to error."""
        self.status = EpisodeStatus.FAILED
        self.end_time = datetime.now()
        self.outcome = EpisodeOutcome.ERROR
        self.episode_data["error_message"] = error_message

        logger.error(f"Failed episode {self.episode_id.value}: {error_message}")

    def get_duration(self) -> Optional[timedelta]:
        """Get episode duration."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def get_average_reward_per_step(self) -> float:
        """Calculate average reward per step."""
        if self.step_count == 0:
            return 0.0
        return self.total_reward / self.step_count

    def is_successful(self) -> bool:
        """Check if episode was successful."""
        return self.outcome in [EpisodeOutcome.SUCCESS, EpisodeOutcome.PARTIAL_SUCCESS]

    def achieved_target_skill(self) -> bool:
        """Check if episode achieved its target skill."""
        if self.target_skill is None or self.performance_metrics is None:
            return False

        skill_score = self.performance_metrics.skill_scores.get(self.target_skill, 0.0)
        return skill_score >= 0.7  # Threshold for skill achievement

    def get_complexity_score(self) -> float:
        """Calculate complexity score based on commands executed."""
        if not self.motion_commands:
            return 0.0

        total_complexity = sum(
            cmd.get_complexity_score() for cmd in self.motion_commands
        )
        return total_complexity / len(self.motion_commands)

    def update_trajectory(self, trajectory: MovementTrajectory) -> None:
        """Update movement trajectory for episode."""
        self.movement_trajectory = trajectory

        # Store trajectory quality metrics
        if "trajectory_metrics" not in self.episode_data:
            self.episode_data["trajectory_metrics"] = {}

        self.episode_data["trajectory_metrics"].update(
            {
                "total_distance": trajectory.get_total_distance(),
                "average_velocity": trajectory.get_average_velocity(),
                "smoothness_score": trajectory.get_smoothness_score(),
            }
        )


class StageType(Enum):
    """Types of curriculum stages."""

    FOUNDATION = "foundation"
    SKILL_BUILDING = "skill_building"
    INTEGRATION = "integration"
    MASTERY = "mastery"
    ADVANCED = "advanced"


class AdvancementCriteria(Enum):
    """Criteria for advancing to next stage."""

    EPISODE_COUNT = "episode_count"
    SUCCESS_RATE = "success_rate"
    SKILL_MASTERY = "skill_mastery"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    TIME_BASED = "time_based"
    MANUAL_APPROVAL = "manual_approval"


@dataclass
class CurriculumStage:
    """
    Entity representing a stage in the curriculum progression.

    Each stage defines learning objectives, success criteria, and advancement rules.
    """

    stage_id: str
    name: str
    stage_type: StageType
    order: int  # Position in curriculum sequence

    # Learning objectives
    target_skills: Set[SkillType] = field(default_factory=set)
    prerequisite_skills: Set[SkillType] = field(default_factory=set)

    # Stage configuration
    difficulty_level: float = 1.0  # 0.0 to 5.0
    expected_duration_episodes: int = 50
    target_success_rate: float = 0.7

    # Advancement criteria
    advancement_criteria: Dict[AdvancementCriteria, float] = field(default_factory=dict)
    min_episodes: int = 10

    # Performance tracking
    episodes_completed: int = 0
    successful_episodes: int = 0
    skill_assessments: Dict[SkillType, SkillAssessment] = field(default_factory=dict)

    # Stage metadata
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def can_advance(self) -> bool:
        """Check if stage advancement criteria are met."""
        # Check minimum episodes
        if self.episodes_completed < self.min_episodes:
            return False

        # Check success rate
        current_success_rate = self.get_success_rate()
        if current_success_rate < self.target_success_rate:
            return False

        # Check skill mastery requirements
        for skill in self.target_skills:
            if not self.is_skill_mastered(skill):
                return False

        # Check additional advancement criteria
        for criteria, threshold in self.advancement_criteria.items():
            if not self._evaluate_criteria(criteria, threshold):
                return False

        return True

    def add_episode_result(self, episode: LearningEpisode) -> None:
        """Add episode result to stage tracking."""
        self.episodes_completed += 1

        if episode.is_successful():
            self.successful_episodes += 1

        # Update skill assessments if available
        if episode.performance_metrics and episode.target_skill:
            skill_score = episode.performance_metrics.skill_scores.get(
                episode.target_skill, 0.0
            )

            # Create skill assessment
            skill = LocomotionSkill(
                skill_type=episode.target_skill,
                mastery_level=MasteryLevel.from_score(skill_score),
                proficiency_score=skill_score,
            )

            assessment = SkillAssessment(
                skill=skill,
                assessment_score=skill_score,
                confidence_level=0.8,  # Could be calculated from episode quality
                evidence_quality=0.8,  # Could be calculated from trajectory quality
            )

            self.skill_assessments[episode.target_skill] = assessment

    def get_success_rate(self) -> float:
        """Calculate current success rate."""
        if self.episodes_completed == 0:
            return 0.0
        return self.successful_episodes / self.episodes_completed

    def is_skill_mastered(
        self, skill: SkillType, threshold: MasteryLevel = MasteryLevel.INTERMEDIATE
    ) -> bool:
        """Check if a skill is mastered at required level."""
        assessment = self.skill_assessments.get(skill)
        if assessment is None:
            return False

        return (
            assessment.skill.mastery_level.get_numeric_value()
            >= threshold.get_numeric_value()
        )

    def get_progress_percentage(self) -> float:
        """Calculate stage progress as percentage."""
        if self.expected_duration_episodes == 0:
            return 100.0 if self.can_advance() else 0.0

        episode_progress = (
            min(self.episodes_completed / self.expected_duration_episodes, 1.0) * 50.0
        )  # Episodes contribute 50%

        success_progress = (
            min(self.get_success_rate() / self.target_success_rate, 1.0) * 30.0
        )  # Success rate contributes 30%

        skill_progress = (
            self._calculate_skill_progress() * 20.0
        )  # Skills contribute 20%

        return episode_progress + success_progress + skill_progress

    def get_remaining_requirements(self) -> Dict[str, Any]:
        """Get remaining requirements for stage advancement."""
        requirements = {}

        # Episode requirements
        if self.episodes_completed < self.min_episodes:
            requirements["min_episodes"] = self.min_episodes - self.episodes_completed

        # Success rate requirements
        current_success_rate = self.get_success_rate()
        if current_success_rate < self.target_success_rate:
            requirements["success_rate_gap"] = (
                self.target_success_rate - current_success_rate
            )

        # Skill mastery requirements
        unmastered_skills = []
        for skill in self.target_skills:
            if not self.is_skill_mastered(skill):
                unmastered_skills.append(skill)

        if unmastered_skills:
            requirements["unmastered_skills"] = unmastered_skills

        return requirements

    def _evaluate_criteria(
        self, criteria: AdvancementCriteria, threshold: float
    ) -> bool:
        """Evaluate specific advancement criteria."""
        if criteria == AdvancementCriteria.EPISODE_COUNT:
            return self.episodes_completed >= threshold
        elif criteria == AdvancementCriteria.SUCCESS_RATE:
            return self.get_success_rate() >= threshold
        elif criteria == AdvancementCriteria.PERFORMANCE_THRESHOLD:
            # Could evaluate average performance metrics
            return True  # Placeholder
        elif criteria == AdvancementCriteria.SKILL_MASTERY:
            # Check percentage of skills mastered
            if not self.target_skills:
                return True

            mastered_count = sum(
                1 for skill in self.target_skills if self.is_skill_mastered(skill)
            )
            mastery_percentage = mastered_count / len(self.target_skills)
            return mastery_percentage >= threshold

        return True  # Default to true for unimplemented criteria

    def _calculate_skill_progress(self) -> float:
        """Calculate overall skill mastery progress."""
        if not self.target_skills:
            return 1.0

        total_progress = 0.0
        for skill in self.target_skills:
            assessment = self.skill_assessments.get(skill)
            if assessment:
                total_progress += assessment.get_adjusted_score()
            # Unassessed skills contribute 0

        return total_progress / len(self.target_skills)
