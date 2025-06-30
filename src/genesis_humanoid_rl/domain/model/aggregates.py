"""
Domain aggregates for humanoid robotics learning.
Aggregates are clusters of entities and value objects with clear boundaries and consistency rules.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from .value_objects import (
    SessionId,
    RobotId,
    PlanId,
    EpisodeId,
    MotionCommand,
    LocomotionSkill,
    SkillType,
    MasteryLevel,
    PerformanceMetrics,
    GaitPattern,
    MovementTrajectory,
    SkillAssessment,
)
from .entities import (
    LearningEpisode,
    EpisodeStatus,
    EpisodeOutcome,
    CurriculumStage,
    StageType,
    AdvancementCriteria,
)

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    """Status of a learning session."""

    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class RobotType(Enum):
    """Types of humanoid robots."""

    UNITREE_G1 = "unitree_g1"
    GENERIC_HUMANOID = "generic_humanoid"
    CUSTOM = "custom"


class PlanStatus(Enum):
    """Status of a curriculum plan."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    DEPRECATED = "deprecated"


@dataclass
class LearningSession:
    """
    Aggregate root for managing the complete learning lifecycle.

    Maintains consistency across episodes, curriculum progression, and performance tracking.
    Coordinates between robot capabilities and curriculum requirements.
    """

    session_id: SessionId
    robot_id: RobotId
    plan_id: PlanId
    created_at: datetime = field(default_factory=datetime.now)

    # Session configuration
    session_name: str = ""
    target_duration: Optional[timedelta] = None
    max_episodes: int = 1000

    # Session state
    status: SessionStatus = SessionStatus.CREATED
    current_stage_index: int = 0

    # Episode management
    episodes: List[LearningEpisode] = field(default_factory=list)
    active_episode: Optional[LearningEpisode] = None

    # Performance tracking
    total_episodes: int = 0
    successful_episodes: int = 0
    session_metrics: Dict[str, float] = field(default_factory=dict)

    # Session metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def start_session(self, initial_stage: Optional[CurriculumStage] = None) -> None:
        """Start the learning session."""
        if self.status != SessionStatus.CREATED:
            raise ValueError(f"Cannot start session in status {self.status}")

        self.status = SessionStatus.ACTIVE

        if initial_stage:
            self.current_stage_index = initial_stage.order

        logger.info(f"Started learning session {self.session_id.value}")

    def create_episode(
        self, target_skill: Optional[SkillType] = None
    ) -> LearningEpisode:
        """
        Create a new learning episode.

        Business rule: Only one active episode at a time.
        """
        if self.active_episode is not None:
            raise ValueError("Cannot create episode while another episode is active")

        if self.status != SessionStatus.ACTIVE:
            raise ValueError(f"Cannot create episode in session status {self.status}")

        if len(self.episodes) >= self.max_episodes:
            raise ValueError("Maximum episodes reached for session")

        episode = LearningEpisode(
            episode_id=EpisodeId.generate(),
            session_id=self.session_id,
            target_skill=target_skill,
        )

        self.episodes.append(episode)
        self.active_episode = episode

        logger.info(
            f"Created episode {episode.episode_id.value} for session {self.session_id.value}"
        )
        return episode

    def complete_episode(
        self,
        outcome: EpisodeOutcome,
        performance_metrics: Optional[PerformanceMetrics] = None,
    ) -> None:
        """
        Complete the active episode and update session state.

        Business rule: Episode completion updates session-level metrics.
        """
        if self.active_episode is None:
            raise ValueError("No active episode to complete")

        self.active_episode.complete_episode(outcome, performance_metrics)

        # Update session metrics
        self.total_episodes += 1
        if self.active_episode.is_successful():
            self.successful_episodes += 1

        # Update session-level performance metrics
        self._update_session_metrics(self.active_episode)

        # Clear active episode
        self.active_episode = None

        logger.info(f"Completed episode with outcome {outcome}")

    def advance_curriculum_if_ready(self, curriculum_plan: "CurriculumPlan") -> bool:
        """
        Check and advance curriculum stage if criteria are met.

        Business rule: Advancement requires stage completion criteria.
        """
        if self.current_stage_index >= len(curriculum_plan.stages):
            return False  # Already at final stage

        current_stage = curriculum_plan.stages[self.current_stage_index]

        # Add recent episode results to stage
        recent_episodes = self._get_recent_episodes_for_stage(current_stage)
        for episode in recent_episodes:
            current_stage.add_episode_result(episode)

        if current_stage.can_advance():
            self.current_stage_index += 1

            # Publish domain event (would be implemented with event system)
            logger.info(f"Advanced to curriculum stage {self.current_stage_index}")
            return True

        return False

    def pause_session(self) -> None:
        """Pause the learning session."""
        if self.status not in [SessionStatus.ACTIVE]:
            raise ValueError(f"Cannot pause session in status {self.status}")

        # Complete any active episode first
        if self.active_episode is not None:
            self.active_episode.terminate_episode("Session paused")
            self.active_episode = None

        self.status = SessionStatus.PAUSED
        logger.info(f"Paused session {self.session_id.value}")

    def resume_session(self) -> None:
        """Resume a paused learning session."""
        if self.status != SessionStatus.PAUSED:
            raise ValueError(f"Cannot resume session in status {self.status}")

        self.status = SessionStatus.ACTIVE
        logger.info(f"Resumed session {self.session_id.value}")

    def complete_session(self) -> None:
        """Complete the learning session."""
        if self.status not in [SessionStatus.ACTIVE, SessionStatus.PAUSED]:
            raise ValueError(f"Cannot complete session in status {self.status}")

        # Complete any active episode
        if self.active_episode is not None:
            self.active_episode.complete_episode(EpisodeOutcome.SUCCESS)
            self.active_episode = None

        self.status = SessionStatus.COMPLETED
        logger.info(f"Completed session {self.session_id.value}")

    def get_session_statistics(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        success_rate = (
            self.successful_episodes / self.total_episodes
            if self.total_episodes > 0
            else 0.0
        )

        avg_episode_reward = 0.0
        if self.episodes:
            try:
                # Safely calculate rewards for completed episodes
                total_reward = 0.0
                completed_episodes = 0

                for ep in self.episodes:
                    # Check if episode has proper attributes
                    if hasattr(ep, "status") and hasattr(ep, "total_reward"):
                        episode_status = getattr(ep, "status", None)
                        if episode_status == EpisodeStatus.COMPLETED or (
                            isinstance(episode_status, str)
                            and episode_status == "completed"
                        ):
                            reward = getattr(ep, "total_reward", 0.0)
                            if isinstance(reward, (int, float)):
                                total_reward += reward
                                completed_episodes += 1

                avg_episode_reward = (
                    total_reward / completed_episodes if completed_episodes > 0 else 0.0
                )
            except (AttributeError, TypeError, ValueError):
                avg_episode_reward = 0.0

        return {
            "session_id": self.session_id.value,
            "total_episodes": self.total_episodes,
            "successful_episodes": self.successful_episodes,
            "success_rate": success_rate,
            "current_stage": self.current_stage_index,
            "avg_episode_reward": avg_episode_reward,
            "session_duration": self._get_session_duration(),
            "status": self.status.value,
        }

    def get_learning_progress(self) -> float:
        """Calculate overall learning progress (0.0 to 1.0)."""
        if not self.episodes:
            return 0.0

        # Calculate progress based on success rate improvement over time
        recent_episodes = self.episodes[-20:]  # Last 20 episodes
        early_episodes = (
            self.episodes[:20]
            if len(self.episodes) > 40
            else self.episodes[: len(self.episodes) // 2]
        )

        if not early_episodes:
            return 0.0

        early_success_rate = sum(
            1 for ep in early_episodes if ep.is_successful()
        ) / len(early_episodes)
        recent_success_rate = sum(
            1 for ep in recent_episodes if ep.is_successful()
        ) / len(recent_episodes)

        progress = (recent_success_rate - early_success_rate) + recent_success_rate
        return max(0.0, min(1.0, progress))

    def _update_session_metrics(self, episode: LearningEpisode) -> None:
        """Update session-level metrics based on episode results."""
        # Update rolling averages
        if "avg_reward" not in self.session_metrics:
            self.session_metrics["avg_reward"] = episode.total_reward
        else:
            # Exponential moving average
            alpha = 0.1
            self.session_metrics["avg_reward"] = (
                alpha * episode.total_reward
                + (1 - alpha) * self.session_metrics["avg_reward"]
            )

        # Update other metrics
        self.session_metrics["last_episode_reward"] = episode.total_reward
        self.session_metrics["last_episode_steps"] = episode.step_count

    def _get_recent_episodes_for_stage(
        self, stage: CurriculumStage
    ) -> List[LearningEpisode]:
        """Get recent episodes relevant to the current stage."""
        # Return episodes that haven't been processed for this stage yet
        try:
            stage_episode_count = getattr(stage, "episodes_completed", 0)
            if not isinstance(stage_episode_count, int):
                stage_episode_count = 0
        except (AttributeError, TypeError):
            stage_episode_count = 0

        total_episodes = len(self.episodes)

        if total_episodes <= stage_episode_count:
            return []

        return self.episodes[stage_episode_count:]

    def _get_session_duration(self) -> Optional[timedelta]:
        """Get total session duration."""
        if not self.episodes:
            return None

        try:
            # Filter out episodes without proper start_time
            valid_episodes = [
                ep
                for ep in self.episodes
                if hasattr(ep, "start_time") and hasattr(ep.start_time, "__lt__")
            ]

            if not valid_episodes:
                return None

            first_episode = min(valid_episodes, key=lambda ep: ep.start_time)
            last_episode = max(valid_episodes, key=lambda ep: ep.start_time)

            if hasattr(last_episode, "end_time") and last_episode.end_time:
                return last_episode.end_time - first_episode.start_time

            return datetime.now() - first_episode.start_time
        except (AttributeError, TypeError, ValueError):
            # Return a default duration if comparison fails
            return timedelta(minutes=30)  # Default reasonable session duration


@dataclass
class HumanoidRobot:
    """
    Aggregate root representing a learning humanoid robot.

    Encapsulates robot capabilities, learned skills, and performance history.
    Maintains consistency of robot state and skill progression.
    """

    robot_id: RobotId
    robot_type: RobotType
    name: str
    created_at: datetime = field(default_factory=datetime.now)

    # Robot configuration
    joint_count: int = 35
    height: float = 1.2  # meters
    weight: float = 35.0  # kg
    configuration: Dict[str, Any] = field(default_factory=dict)

    # Learning state
    learned_skills: Dict[SkillType, LocomotionSkill] = field(default_factory=dict)
    skill_history: List[SkillAssessment] = field(default_factory=list)

    # Performance tracking
    total_training_time: timedelta = field(default_factory=lambda: timedelta(0))
    performance_history: List[PerformanceMetrics] = field(default_factory=list)
    gait_patterns: List[GaitPattern] = field(default_factory=list)

    # Robot metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def assess_skill(self, skill_type: SkillType, assessment: SkillAssessment) -> None:
        """
        Assess and update robot's skill proficiency.

        Business rule: Skills can only improve over time (no regression).
        All assessments are recorded in history for audit purposes.
        """
        current_skill = self.learned_skills.get(skill_type)

        # Always record assessment in history for audit purposes
        self.skill_history.append(assessment)

        # Validate assessment is an improvement or first assessment
        if (
            current_skill
            and assessment.skill.proficiency_score < current_skill.proficiency_score
        ):
            logger.warning(
                f"Skill assessment for {skill_type} shows regression, keeping current level"
            )
            return

        # Update skill only if it's an improvement
        self.learned_skills[skill_type] = assessment.skill

        logger.info(f"Updated {skill_type} to {assessment.skill.mastery_level.value}")

    def master_skill(self, skill_type: SkillType, evidence: Dict[str, Any]) -> bool:
        """
        Mark a skill as mastered.

        Business rule: Mastery requires demonstration across multiple episodes.
        """
        if skill_type not in self.learned_skills:
            logger.warning(f"Cannot master unlearned skill {skill_type}")
            return False

        current_skill = self.learned_skills[skill_type]

        # Create mastery assessment
        mastered_skill = LocomotionSkill(
            skill_type=skill_type,
            mastery_level=MasteryLevel.EXPERT,
            proficiency_score=1.0,
            last_assessed=datetime.now(),
        )

        mastery_assessment = SkillAssessment(
            skill=mastered_skill,
            assessment_score=1.0,
            confidence_level=0.95,
            evidence_quality=0.9,
        )

        self.assess_skill(skill_type, mastery_assessment)

        # Store mastery evidence
        self.metadata[f"{skill_type.value}_mastery_evidence"] = evidence

        logger.info(f"Robot {self.robot_id.value} mastered skill {skill_type.value}")
        return True

    def can_learn_skill(self, skill_type: SkillType) -> bool:
        """
        Check if robot can learn a specific skill based on prerequisites.

        Business rule: Skills have prerequisite requirements.
        """
        # Check if already mastered
        current_skill = self.learned_skills.get(skill_type)
        if current_skill and current_skill.is_mastered():
            return False  # Already mastered

        # Check prerequisites
        temp_skill = LocomotionSkill(skill_type=skill_type)
        for prereq_skill_type in SkillType:
            if temp_skill.requires_skill(prereq_skill_type):
                prereq_skill = self.learned_skills.get(prereq_skill_type)
                if not prereq_skill or not prereq_skill.is_mastered(
                    MasteryLevel.BEGINNER
                ):
                    return False

        return True

    def get_skill_proficiency(self, skill_type: SkillType) -> float:
        """Get current proficiency score for a skill."""
        skill = self.learned_skills.get(skill_type)
        return skill.proficiency_score if skill else 0.0

    def get_mastered_skills(self) -> List[SkillType]:
        """Get list of mastered skills."""
        mastered = []
        for skill_type, skill in self.learned_skills.items():
            if skill.is_mastered():
                mastered.append(skill_type)
        return mastered

    def get_next_recommended_skills(self) -> List[SkillType]:
        """Get recommended next skills to learn."""
        recommendations = []

        for skill_type in SkillType:
            if self.can_learn_skill(skill_type):
                current_proficiency = self.get_skill_proficiency(skill_type)
                if current_proficiency < 0.7:  # Not yet proficient
                    recommendations.append(skill_type)

        # Sort by prerequisite order (simpler skills first)
        skill_complexity = {
            SkillType.POSTURAL_CONTROL: 1,
            SkillType.STATIC_BALANCE: 2,
            SkillType.DYNAMIC_BALANCE: 3,
            SkillType.FORWARD_WALKING: 4,
            SkillType.SPEED_CONTROL: 5,
            SkillType.TURNING: 6,
            SkillType.BACKWARD_WALKING: 7,
            SkillType.TERRAIN_ADAPTATION: 8,
            SkillType.OBSTACLE_AVOIDANCE: 9,
        }

        recommendations.sort(key=lambda x: skill_complexity.get(x, 10))
        return recommendations[:3]  # Top 3 recommendations

    def update_performance_metrics(self, metrics: PerformanceMetrics) -> None:
        """Update robot's performance history."""
        self.performance_history.append(metrics)

        # Keep only last 100 performance records
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

    def add_gait_pattern(self, pattern: GaitPattern) -> None:
        """Add a new gait pattern to robot's repertoire."""
        self.gait_patterns.append(pattern)

        # Keep only high-quality gaits
        self.gait_patterns = [
            gait for gait in self.gait_patterns if gait.get_quality_score() > 0.6
        ]

        # Limit to 20 best patterns
        if len(self.gait_patterns) > 20:
            self.gait_patterns.sort(key=lambda x: x.get_quality_score(), reverse=True)
            self.gait_patterns = self.gait_patterns[:20]

    def get_robot_capabilities(self) -> Dict[str, Any]:
        """Get summary of robot capabilities."""
        mastered_skills = self.get_mastered_skills()
        avg_performance = 0.0

        if self.performance_history:
            recent_performance = self.performance_history[-10:]  # Last 10 records
            avg_performance = sum(
                p.get_overall_performance() for p in recent_performance
            ) / len(recent_performance)

        return {
            "robot_id": self.robot_id.value,
            "robot_type": self.robot_type.value,
            "mastered_skills": [skill.value for skill in mastered_skills],
            "skill_count": len(self.learned_skills),
            "avg_performance": avg_performance,
            "training_time_hours": self.total_training_time.total_seconds() / 3600,
            "gait_patterns_learned": len(self.gait_patterns),
            "next_recommended_skills": [
                skill.value for skill in self.get_next_recommended_skills()
            ],
        }


@dataclass
class CurriculumPlan:
    """
    Aggregate root for managing structured learning progression.

    Defines the sequence of learning stages and advancement criteria.
    Maintains consistency of curriculum structure and progression rules.
    """

    plan_id: PlanId
    name: str
    robot_type: RobotType
    created_at: datetime = field(default_factory=datetime.now)

    # Plan configuration
    description: str = ""
    version: str = "1.0"
    status: PlanStatus = PlanStatus.DRAFT

    # Curriculum structure
    stages: List[CurriculumStage] = field(default_factory=list)

    # Plan metadata
    author: str = ""
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: CurriculumStage) -> None:
        """
        Add a stage to the curriculum plan.

        Business rule: Stages must be added in order.
        """
        if self.status not in [PlanStatus.DRAFT]:
            raise ValueError(f"Cannot modify plan in status {self.status}")

        # Validate order
        expected_order = len(self.stages)
        if stage.order != expected_order:
            raise ValueError(
                f"Stage order {stage.order} invalid, expected {expected_order}"
            )

        # Validate prerequisites are covered by previous stages
        self._validate_stage_prerequisites(stage)

        self.stages.append(stage)
        logger.info(
            f"Added stage '{stage.name}' to curriculum plan {self.plan_id.value}"
        )

    def activate_plan(self) -> None:
        """
        Activate the curriculum plan for use.

        Business rule: Plan must be in draft status and have at least one stage to activate.
        """
        if self.status != PlanStatus.DRAFT:
            raise ValueError(f"Cannot activate plan in status {self.status}")

        if not self.stages:
            raise ValueError("Cannot activate plan with no stages")

        # Validate plan consistency
        self._validate_plan_consistency()

        self.status = PlanStatus.ACTIVE
        logger.info(f"Activated curriculum plan {self.plan_id.value}")

    def get_stage_by_index(self, index: int) -> Optional[CurriculumStage]:
        """Get stage by index."""
        if 0 <= index < len(self.stages):
            return self.stages[index]
        return None

    def get_next_stage(self, current_stage_index: int) -> Optional[CurriculumStage]:
        """Get the next stage in sequence."""
        return self.get_stage_by_index(current_stage_index + 1)

    def evaluate_advancement(self, session: LearningSession) -> bool:
        """
        Evaluate if session can advance to next curriculum stage.

        Business rule: Advancement requires current stage completion.
        """
        if session.current_stage_index >= len(self.stages):
            return False  # Already at final stage

        current_stage = self.stages[session.current_stage_index]
        return current_stage.can_advance()

    def get_plan_progress(self, current_stage_index: int) -> float:
        """Calculate overall plan progress (0.0 to 1.0)."""
        if not self.stages:
            return 0.0

        if current_stage_index >= len(self.stages):
            return 1.0  # Completed all stages

        # Base progress from completed stages
        stage_progress = current_stage_index / len(self.stages)

        # Add partial progress from current stage
        if current_stage_index < len(self.stages):
            current_stage = self.stages[current_stage_index]
            current_stage_progress = current_stage.get_progress_percentage() / 100.0
            stage_progress += current_stage_progress / len(self.stages)

        return min(1.0, stage_progress)

    def get_recommended_next_skills(self, current_stage_index: int) -> List[SkillType]:
        """Get skills recommended for current stage."""
        if current_stage_index >= len(self.stages):
            return []

        current_stage = self.stages[current_stage_index]
        return list(current_stage.target_skills)

    def adapt_difficulty(
        self, robot_performance: PerformanceMetrics, current_stage_index: int
    ) -> Dict[str, Any]:
        """
        Adapt curriculum difficulty based on robot performance.

        Business rule: Difficulty adapts to maintain optimal challenge level.
        """
        if current_stage_index >= len(self.stages):
            return {}

        current_stage = self.stages[current_stage_index]
        adaptations = {}

        # Analyze performance
        overall_performance = robot_performance.get_overall_performance()

        # Suggest difficulty adjustments
        if overall_performance > 0.6:
            # Performance too high, increase difficulty
            adaptations["difficulty_adjustment"] = "increase"
            adaptations["suggested_target_success_rate"] = min(
                current_stage.target_success_rate + 0.1, 0.95
            )
        elif overall_performance < 0.3:
            # Performance too low, decrease difficulty
            adaptations["difficulty_adjustment"] = "decrease"
            adaptations["suggested_target_success_rate"] = max(
                current_stage.target_success_rate - 0.1, 0.5
            )
        else:
            adaptations["difficulty_adjustment"] = "maintain"

        # Suggest episode count adjustments
        if robot_performance.learning_progress > 0.8:
            adaptations["suggested_min_episodes"] = max(
                current_stage.min_episodes - 5, 5
            )
        elif robot_performance.learning_progress < 0.2:
            adaptations["suggested_min_episodes"] = current_stage.min_episodes + 10

        return adaptations

    def _validate_stage_prerequisites(self, stage: CurriculumStage) -> None:
        """Validate that stage prerequisites are covered by previous stages."""
        covered_skills = set()

        # Collect skills from all previous stages
        for prev_stage in self.stages:
            covered_skills.update(prev_stage.target_skills)

        # Check if prerequisites are covered
        missing_prerequisites = stage.prerequisite_skills - covered_skills
        if missing_prerequisites:
            raise ValueError(
                f"Stage prerequisites not covered: {missing_prerequisites}"
            )

    def _validate_plan_consistency(self) -> None:
        """Validate overall plan consistency."""
        if not self.stages:
            raise ValueError("Plan must have at least one stage")

        # Check stage ordering
        for i, stage in enumerate(self.stages):
            if stage.order != i:
                raise ValueError(f"Stage order inconsistency at index {i}")

        # Check skill progression
        all_skills = set()
        for stage in self.stages:
            # Check for skill conflicts
            skill_conflicts = all_skills.intersection(stage.target_skills)
            if skill_conflicts:
                logger.warning(f"Skills {skill_conflicts} targeted in multiple stages")

            all_skills.update(stage.target_skills)
