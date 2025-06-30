"""
Repository interfaces for domain persistence.
Define contracts for data access without implementation details.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from .model.value_objects import SessionId, RobotId, PlanId, SkillType
from .model.aggregates import LearningSession, HumanoidRobot, CurriculumPlan


@dataclass
class DomainEvent:
    """Domain event for event sourcing."""

    event_type: str
    aggregate_id: str
    payload: Dict[str, Any]
    occurred_at: float


class LearningSessionRepository(ABC):
    """Repository interface for learning session persistence."""

    @abstractmethod
    def save(self, session: LearningSession) -> None:
        """Save a learning session."""
        pass

    @abstractmethod
    def find_by_id(self, session_id: SessionId) -> Optional[LearningSession]:
        """Find session by ID."""
        pass

    @abstractmethod
    def find_active_sessions(self) -> List[LearningSession]:
        """Find all active learning sessions."""
        pass

    @abstractmethod
    def find_by_robot_id(self, robot_id: RobotId) -> List[LearningSession]:
        """Find sessions for a specific robot."""
        pass

    @abstractmethod
    def find_completed_sessions(
        self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None
    ) -> List[LearningSession]:
        """Find completed sessions within date range."""
        pass


class HumanoidRobotRepository(ABC):
    """Repository interface for robot persistence."""

    @abstractmethod
    def save(self, robot: HumanoidRobot) -> None:
        """Save a robot."""
        pass

    @abstractmethod
    def find_by_id(self, robot_id: RobotId) -> Optional[HumanoidRobot]:
        """Find robot by ID."""
        pass

    @abstractmethod
    def find_all(self) -> List[HumanoidRobot]:
        """Find all robots."""
        pass

    @abstractmethod
    def find_by_skill_mastery(
        self, skill: SkillType, min_proficiency: float = 0.7
    ) -> List[HumanoidRobot]:
        """Find robots that have mastered a specific skill."""
        pass


class CurriculumPlanRepository(ABC):
    """Repository interface for curriculum plan persistence."""

    @abstractmethod
    def save(self, plan: CurriculumPlan) -> None:
        """Save a curriculum plan."""
        pass

    @abstractmethod
    def find_by_id(self, plan_id: PlanId) -> Optional[CurriculumPlan]:
        """Find plan by ID."""
        pass

    @abstractmethod
    def find_active_plans(self) -> List[CurriculumPlan]:
        """Find all active curriculum plans."""
        pass

    @abstractmethod
    def find_by_robot_type(self, robot_type: str) -> List[CurriculumPlan]:
        """Find plans for specific robot type."""
        pass


class DomainEventRepository(ABC):
    """Repository interface for domain event persistence."""

    @abstractmethod
    def save(self, event: DomainEvent) -> None:
        """Save a domain event."""
        pass

    @abstractmethod
    def find_by_aggregate(self, aggregate_id: str) -> List[DomainEvent]:
        """Find events for specific aggregate."""
        pass

    @abstractmethod
    def find_by_type(self, event_type: str, limit: int = 100) -> List[DomainEvent]:
        """Find events of specific type."""
        pass

    @abstractmethod
    def find_recent_events(self, limit: int = 100) -> List[DomainEvent]:
        """Find recent events."""
        pass
