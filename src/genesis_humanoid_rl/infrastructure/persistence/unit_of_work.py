"""
Unit of Work pattern implementation for managing transaction boundaries.

Ensures atomicity across multiple repository operations within a single business transaction.
Addresses the transaction boundary mismatch identified in Five Whys analysis.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Generator, Set

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


class UnitOfWorkError(Exception):
    """Exception raised for Unit of Work transaction failures."""

    pass


class AbstractUnitOfWork(ABC):
    """Abstract Unit of Work for managing transaction boundaries."""

    sessions: LearningSessionRepository
    robots: HumanoidRobotRepository
    plans: CurriculumPlanRepository
    events: DomainEventRepository

    def __enter__(self):
        """Enter the transaction context."""
        return self

    def __exit__(self, *args):
        """Exit the transaction context with proper cleanup."""
        self.rollback()

    def commit(self) -> None:
        """Commit all changes in the transaction."""
        self._commit()
        self._clear_identity_map()

    @abstractmethod
    def _commit(self) -> None:
        """Implementation-specific commit logic."""
        pass

    @abstractmethod
    def rollback(self) -> None:
        """Rollback all changes in the transaction."""
        pass

    def _clear_identity_map(self) -> None:
        """Clear any cached entities after transaction completion."""
        # Can be implemented by subclasses if needed
        pass


class SQLiteUnitOfWork(AbstractUnitOfWork):
    """SQLite implementation of Unit of Work pattern."""

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize Unit of Work with database connection.

        Args:
            db_connection: Database connection manager
        """
        self.db_connection = db_connection
        self._transaction_active = False
        self._connection = None

        # Initialize repositories with this UoW instance
        # They will check for active transaction
        self.sessions = SQLiteLearningSessionRepository(db_connection)
        self.robots = SQLiteHumanoidRobotRepository(db_connection)
        self.plans = SQLiteCurriculumPlanRepository(db_connection)
        self.events = SQLiteDomainEventRepository(db_connection)

        # Set UoW reference in repositories
        self.sessions._unit_of_work = self
        self.robots._unit_of_work = self
        self.plans._unit_of_work = self
        self.events._unit_of_work = self

        # Track entities for identity map (if needed in future)
        self._identity_map: Dict[str, Any] = {}
        self._new_entities: Set[Any] = set()
        self._dirty_entities: Set[Any] = set()
        self._removed_entities: Set[Any] = set()

    def __enter__(self):
        """Start a new transaction."""
        try:
            self._connection = self.db_connection._get_connection()
            self._connection.execute("BEGIN")
            self._transaction_active = True
            logger.debug("Started Unit of Work transaction")
            return self
        except Exception as e:
            logger.error(f"Failed to start Unit of Work transaction: {e}")
            raise UnitOfWorkError(f"Transaction start failed: {e}") from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End transaction with commit or rollback."""
        if self._transaction_active:
            if exc_type is not None:
                logger.warning(
                    f"Transaction rolling back due to exception: {exc_type.__name__}: {exc_val}"
                )
                self.rollback()
            else:
                try:
                    self.commit()
                except Exception as e:
                    logger.error(f"Commit failed, rolling back: {e}")
                    self.rollback()
                    raise

        # Always ensure transaction is closed
        if self._connection:
            self._connection = None

    def _commit(self) -> None:
        """Commit the current transaction."""
        if not self._transaction_active:
            raise UnitOfWorkError("No active transaction to commit")

        try:
            self._connection.execute("COMMIT")
            self._transaction_active = False
            logger.debug("Unit of Work transaction committed successfully")
        except Exception as e:
            logger.error(f"Transaction commit failed: {e}")
            self.rollback()
            raise UnitOfWorkError(f"Commit failed: {e}") from e

    def rollback(self) -> None:
        """Rollback the current transaction."""
        if self._transaction_active and self._connection:
            try:
                self._connection.execute("ROLLBACK")
                self._transaction_active = False
                logger.debug("Unit of Work transaction rolled back")
            except Exception as e:
                logger.error(f"Transaction rollback failed: {e}")
                # Don't raise here as we're already in error handling

        self._clear_identity_map()

    def _clear_identity_map(self) -> None:
        """Clear the identity map after transaction completion."""
        self._identity_map.clear()
        self._new_entities.clear()
        self._dirty_entities.clear()
        self._removed_entities.clear()

    def track_new(self, entity: Any) -> None:
        """Track a new entity for persistence.

        Args:
            entity: Domain entity to track as new
        """
        self._new_entities.add(entity)
        logger.debug(f"Tracking new entity: {type(entity).__name__}")

    def track_dirty(self, entity: Any) -> None:
        """Track an entity as modified.

        Args:
            entity: Domain entity to track as dirty
        """
        if entity not in self._new_entities:
            self._dirty_entities.add(entity)
            logger.debug(f"Tracking dirty entity: {type(entity).__name__}")

    def track_removed(self, entity: Any) -> None:
        """Track an entity for removal.

        Args:
            entity: Domain entity to track for removal
        """
        self._removed_entities.add(entity)
        if entity in self._new_entities:
            self._new_entities.remove(entity)
        if entity in self._dirty_entities:
            self._dirty_entities.remove(entity)
        logger.debug(f"Tracking removed entity: {type(entity).__name__}")


@contextmanager
def unit_of_work(
    db_connection: DatabaseConnection,
) -> Generator[SQLiteUnitOfWork, None, None]:
    """Context manager for Unit of Work transactions.

    Args:
        db_connection: Database connection to use

    Yields:
        Unit of Work instance for the transaction

    Example:
        with unit_of_work(db) as uow:
            # Make multiple repository calls
            session = uow.sessions.find_by_id(session_id)
            robot = uow.robots.find_by_id(session.robot_id)

            # Modify entities
            session.complete_episode(outcome, metrics)
            robot.assess_skill(skill_type, assessment)

            # Save changes
            uow.sessions.save(session)
            uow.robots.save(robot)

            # All committed atomically
            uow.commit()
    """
    uow = SQLiteUnitOfWork(db_connection)
    try:
        with uow:
            yield uow
    except Exception as e:
        logger.error(f"Unit of Work transaction failed: {e}")
        raise
