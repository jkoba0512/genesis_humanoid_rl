"""Factory for creating repository instances."""

import logging
from typing import Optional

from ...domain.repositories import (
    CurriculumPlanRepository,
    DomainEventRepository,
    HumanoidRobotRepository,
    LearningSessionRepository,
)
from .database import DatabaseConnection
from .sqlite_repositories import (
    SQLiteCurriculumPlanRepository,
    SQLiteDomainEventRepository,
    SQLiteHumanoidRobotRepository,
    SQLiteLearningSessionRepository,
)

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating repository instances with shared database connection."""

    def __init__(self, db_path: str = "genesis_humanoid_rl.db"):
        """Initialize repository factory.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_connection = DatabaseConnection(db_path)
        self._session_repo: Optional[LearningSessionRepository] = None
        self._robot_repo: Optional[HumanoidRobotRepository] = None
        self._plan_repo: Optional[CurriculumPlanRepository] = None
        self._event_repo: Optional[DomainEventRepository] = None

        logger.info(f"Repository factory initialized with database: {db_path}")

    def get_session_repository(self) -> LearningSessionRepository:
        """Get learning session repository instance."""
        if self._session_repo is None:
            self._session_repo = SQLiteLearningSessionRepository(self._db_connection)
        return self._session_repo

    def get_robot_repository(self) -> HumanoidRobotRepository:
        """Get humanoid robot repository instance."""
        if self._robot_repo is None:
            self._robot_repo = SQLiteHumanoidRobotRepository(self._db_connection)
        return self._robot_repo

    def get_curriculum_repository(self) -> CurriculumPlanRepository:
        """Get curriculum plan repository instance."""
        if self._plan_repo is None:
            self._plan_repo = SQLiteCurriculumPlanRepository(self._db_connection)
        return self._plan_repo

    def get_event_repository(self) -> DomainEventRepository:
        """Get domain event repository instance."""
        if self._event_repo is None:
            self._event_repo = SQLiteDomainEventRepository(self._db_connection)
        return self._event_repo

    def close(self) -> None:
        """Close database connection and cleanup resources."""
        if self._db_connection:
            self._db_connection.close()
            logger.info("Repository factory closed")
