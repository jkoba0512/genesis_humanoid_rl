"""
Application commands for training orchestration.
Commands represent user intentions and trigger application services.
"""

from dataclasses import dataclass
from typing import Optional

from ..domain.model.value_objects import SessionId, RobotId, PlanId, SkillType


@dataclass(frozen=True)
class StartTrainingSessionCommand:
    """Command to start a new training session."""

    robot_id: RobotId
    plan_id: PlanId
    session_name: str
    max_episodes: int = 100


@dataclass(frozen=True)
class ExecuteEpisodeCommand:
    """Command to execute a training episode."""

    session_id: SessionId
    target_skill: Optional[SkillType] = None
    max_steps: int = 200


@dataclass(frozen=True)
class AdvanceCurriculumCommand:
    """Command to advance curriculum stage."""

    session_id: SessionId
