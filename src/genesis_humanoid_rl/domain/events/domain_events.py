"""
Domain events for humanoid robotics learning.
Events enable loose coupling and provide audit trail of domain changes.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
from datetime import datetime
from enum import Enum

from ..model.value_objects import (
    SessionId,
    RobotId,
    EpisodeId,
    PlanId,
    SkillType,
    MasteryLevel,
    PerformanceMetrics,
)
from ..model.entities import EpisodeOutcome, CurriculumStage


class EventType(Enum):
    """Types of domain events."""

    EPISODE_COMPLETED = "episode_completed"
    CURRICULUM_STAGE_ADVANCED = "curriculum_stage_advanced"
    SKILL_MASTERED = "skill_mastered"
    LEARNING_SESSION_STARTED = "learning_session_started"
    LEARNING_SESSION_COMPLETED = "learning_session_completed"
    PERFORMANCE_MILESTONE_REACHED = "performance_milestone_reached"


@dataclass(frozen=True)
class EpisodeCompleted:
    """Published when a learning episode completes."""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int
    metadata: Dict[str, Any]

    episode_id: EpisodeId
    session_id: SessionId
    robot_id: RobotId
    outcome: EpisodeOutcome
    total_reward: float
    step_count: int
    episode_duration_seconds: float
    performance_metrics: Optional[PerformanceMetrics] = None
    target_skill: Optional[SkillType] = None


@dataclass(frozen=True)
class CurriculumStageAdvanced:
    """Published when robot advances to next curriculum stage."""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int
    metadata: Dict[str, Any]

    session_id: SessionId
    robot_id: RobotId
    plan_id: PlanId
    previous_stage_index: int
    new_stage_index: int
    previous_stage_name: str
    new_stage_name: str
    advancement_criteria_met: Dict[str, Any] = field(default_factory=dict)
    episodes_in_previous_stage: int = 0
    success_rate_in_previous_stage: float = 0.0


@dataclass(frozen=True)
class SkillMastered:
    """Published when robot masters a locomotion skill."""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int
    metadata: Dict[str, Any]

    robot_id: RobotId
    session_id: SessionId
    skill_type: SkillType
    mastery_level: MasteryLevel
    proficiency_score: float
    episodes_to_mastery: int
    training_time_hours: float
    mastery_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LearningSessionStarted:
    """Published when a new learning session begins."""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int
    metadata: Dict[str, Any]

    session_id: SessionId
    robot_id: RobotId
    plan_id: PlanId
    session_name: str
    initial_stage_index: int
    max_episodes: int
    target_skills: List[SkillType] = field(default_factory=list)


@dataclass(frozen=True)
class LearningSessionCompleted:
    """Published when a learning session completes."""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int
    metadata: Dict[str, Any]

    session_id: SessionId
    robot_id: RobotId
    plan_id: PlanId
    final_stage_index: int
    total_episodes: int
    successful_episodes: int
    session_duration_hours: float
    skills_mastered: List[SkillType]
    final_performance_metrics: PerformanceMetrics
    completion_reason: (
        str  # "curriculum_completed", "max_episodes", "manual_stop", etc.
    )


@dataclass(frozen=True)
class PerformanceMilestoneReached:
    """Published when robot reaches a significant performance milestone."""

    event_id: str
    event_type: EventType
    occurred_at: datetime
    aggregate_id: str
    version: int
    metadata: Dict[str, Any]

    robot_id: RobotId
    session_id: SessionId
    milestone_type: (
        str  # "success_rate_threshold", "episode_count", "skill_combination", etc.
    )
    milestone_value: float
    threshold_crossed: float
    performance_metrics: PerformanceMetrics
    contributing_factors: Dict[str, Any] = field(default_factory=dict)


# Event factory functions


def create_episode_completed_event(
    episode_id: EpisodeId,
    session_id: SessionId,
    robot_id: RobotId,
    outcome: EpisodeOutcome,
    total_reward: float,
    step_count: int,
    duration_seconds: float,
    performance_metrics: Optional[PerformanceMetrics] = None,
    target_skill: Optional[SkillType] = None,
) -> EpisodeCompleted:
    """Factory function for episode completed events."""
    import uuid

    return EpisodeCompleted(
        event_id=str(uuid.uuid4()),
        event_type=EventType.EPISODE_COMPLETED,
        occurred_at=datetime.now(),
        aggregate_id=session_id.value,
        version=1,
        metadata={},
        episode_id=episode_id,
        session_id=session_id,
        robot_id=robot_id,
        outcome=outcome,
        total_reward=total_reward,
        step_count=step_count,
        episode_duration_seconds=duration_seconds,
        performance_metrics=performance_metrics,
        target_skill=target_skill,
    )


def create_curriculum_advanced_event(
    session_id: SessionId,
    robot_id: RobotId,
    plan_id: PlanId,
    previous_stage: CurriculumStage,
    new_stage: CurriculumStage,
    advancement_criteria: Dict[str, Any],
) -> CurriculumStageAdvanced:
    """Factory function for curriculum advancement events."""
    import uuid

    return CurriculumStageAdvanced(
        event_id=str(uuid.uuid4()),
        event_type=EventType.CURRICULUM_STAGE_ADVANCED,
        occurred_at=datetime.now(),
        aggregate_id=session_id.value,
        version=1,
        metadata={},
        session_id=session_id,
        robot_id=robot_id,
        plan_id=plan_id,
        previous_stage_index=previous_stage.order,
        new_stage_index=new_stage.order,
        previous_stage_name=previous_stage.name,
        new_stage_name=new_stage.name,
        advancement_criteria_met=advancement_criteria,
        episodes_in_previous_stage=previous_stage.episodes_completed,
        success_rate_in_previous_stage=previous_stage.get_success_rate(),
    )


def create_skill_mastered_event(
    robot_id: RobotId,
    session_id: SessionId,
    skill_type: SkillType,
    mastery_level: MasteryLevel,
    proficiency_score: float,
    episodes_to_mastery: int,
    training_time_hours: float,
    evidence: Dict[str, Any],
) -> SkillMastered:
    """Factory function for skill mastery events."""
    import uuid

    return SkillMastered(
        event_id=str(uuid.uuid4()),
        event_type=EventType.SKILL_MASTERED,
        occurred_at=datetime.now(),
        aggregate_id=robot_id.value,
        version=1,
        metadata={},
        robot_id=robot_id,
        session_id=session_id,
        skill_type=skill_type,
        mastery_level=mastery_level,
        proficiency_score=proficiency_score,
        episodes_to_mastery=episodes_to_mastery,
        training_time_hours=training_time_hours,
        mastery_evidence=evidence,
    )


def create_session_started_event(
    session_id: SessionId,
    robot_id: RobotId,
    plan_id: PlanId,
    session_name: str,
    initial_stage_index: int,
    max_episodes: int,
    target_skills: List[SkillType],
) -> LearningSessionStarted:
    """Factory function for session started events."""
    import uuid

    return LearningSessionStarted(
        event_id=str(uuid.uuid4()),
        event_type=EventType.LEARNING_SESSION_STARTED,
        occurred_at=datetime.now(),
        aggregate_id=session_id.value,
        version=1,
        metadata={},
        session_id=session_id,
        robot_id=robot_id,
        plan_id=plan_id,
        session_name=session_name,
        initial_stage_index=initial_stage_index,
        max_episodes=max_episodes,
        target_skills=target_skills,
    )


def create_session_completed_event(
    session_id: SessionId,
    robot_id: RobotId,
    plan_id: PlanId,
    final_stage_index: int,
    total_episodes: int,
    successful_episodes: int,
    session_duration_hours: float,
    skills_mastered: List[SkillType],
    final_performance_metrics: PerformanceMetrics,
    completion_reason: str,
) -> LearningSessionCompleted:
    """Factory function for session completed events."""
    import uuid

    return LearningSessionCompleted(
        event_id=str(uuid.uuid4()),
        event_type=EventType.LEARNING_SESSION_COMPLETED,
        occurred_at=datetime.now(),
        aggregate_id=session_id.value,
        version=1,
        metadata={},
        session_id=session_id,
        robot_id=robot_id,
        plan_id=plan_id,
        final_stage_index=final_stage_index,
        total_episodes=total_episodes,
        successful_episodes=successful_episodes,
        session_duration_hours=session_duration_hours,
        skills_mastered=skills_mastered,
        final_performance_metrics=final_performance_metrics,
        completion_reason=completion_reason,
    )


def create_milestone_reached_event(
    robot_id: RobotId,
    session_id: SessionId,
    milestone_type: str,
    milestone_value: float,
    threshold_crossed: float,
    performance_metrics: PerformanceMetrics,
    contributing_factors: Dict[str, Any],
) -> PerformanceMilestoneReached:
    """Factory function for milestone reached events."""
    import uuid

    return PerformanceMilestoneReached(
        event_id=str(uuid.uuid4()),
        event_type=EventType.PERFORMANCE_MILESTONE_REACHED,
        occurred_at=datetime.now(),
        aggregate_id=robot_id.value,
        version=1,
        metadata={},
        robot_id=robot_id,
        session_id=session_id,
        milestone_type=milestone_type,
        milestone_value=milestone_value,
        threshold_crossed=threshold_crossed,
        performance_metrics=performance_metrics,
        contributing_factors=contributing_factors,
    )
