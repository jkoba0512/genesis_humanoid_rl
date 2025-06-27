"""
Database-related test fixtures and context managers.

Provides isolated database instances for testing with automatic cleanup
and transaction management.
"""

import tempfile
import pytest
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional
from unittest.mock import Mock

from src.genesis_humanoid_rl.infrastructure.persistence.database import DatabaseConnection
from src.genesis_humanoid_rl.infrastructure.persistence.unit_of_work import SQLiteUnitOfWork
from src.genesis_humanoid_rl.application.services.training_transaction_service import TrainingTransactionService


@contextmanager
def temporary_database(name: Optional[str] = None) -> Generator[Path, None, None]:
    """
    Context manager for creating temporary SQLite databases.
    
    Args:
        name: Optional name for the database file
        
    Yields:
        Path to the temporary database file
        
    Example:
        with temporary_database("test.db") as db_path:
            # Use database
            pass
        # Database file automatically deleted
    """
    suffix = f"_{name}.db" if name else ".db"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        db_path = Path(tmp_file.name)
    
    try:
        yield db_path
    finally:
        # Clean up database file
        if db_path.exists():
            db_path.unlink()


@contextmanager
def database_connection(db_path: Optional[Path] = None) -> Generator[DatabaseConnection, None, None]:
    """
    Context manager for database connections with automatic cleanup.
    
    Args:
        db_path: Optional path to database file. If None, creates temporary database.
        
    Yields:
        DatabaseConnection instance
        
    Example:
        with database_connection() as db:
            # Use database connection
            pass
        # Connection automatically closed
    """
    if db_path is None:
        with temporary_database() as temp_path:
            db = DatabaseConnection(str(temp_path))
            try:
                yield db
            finally:
                db.close()
    else:
        db = DatabaseConnection(str(db_path))
        try:
            yield db
        finally:
            db.close()


@pytest.fixture
def temp_database_path():
    """Pytest fixture for temporary database path."""
    with temporary_database() as db_path:
        yield db_path


@pytest.fixture
def database_connection_fixture(temp_database_path):
    """Pytest fixture for database connection."""
    with database_connection(temp_database_path) as db:
        yield db


@pytest.fixture
def unit_of_work_fixture(database_connection_fixture):
    """Pytest fixture for SQLite Unit of Work."""
    return SQLiteUnitOfWork(database_connection_fixture)


@pytest.fixture
def training_service_fixture(database_connection_fixture):
    """Pytest fixture for training transaction service."""
    return TrainingTransactionService(database_connection_fixture)


@pytest.fixture
def populated_database(database_connection_fixture):
    """
    Pytest fixture for database with sample data.
    
    Creates a database with test robots, sessions, and episodes for testing.
    """
    db = database_connection_fixture
    
    # Create sample data
    sample_data = {
        'robots': [
            {
                'id': 'robot-test-1',
                'type': 'UNITREE_G1',
                'name': 'Test Robot 1',
                'joint_count': 35,
                'height': 1.2,
                'weight': 35.0
            }
        ],
        'sessions': [
            {
                'id': 'session-test-1',
                'robot_id': 'robot-test-1',
                'plan_id': 'plan-test-1',
                'start_time': '2024-01-01T10:00:00',
                'end_time': '2024-01-01T11:00:00'
            }
        ],
        'episodes': [
            {
                'id': 'episode-test-1',
                'session_id': 'session-test-1',
                'step_count': 1000,
                'total_reward': 85.5,
                'outcome': 'SUCCESS'
            }
        ]
    }
    
    # Insert sample data (implementation would depend on your database schema)
    # This is a placeholder - you would implement actual data insertion here
    db._sample_data = sample_data  # Store for test access
    
    return db


class DatabaseTestCase:
    """
    Base test case class for database-related tests.
    
    Provides common setup and utilities for database testing.
    Should be used as a mixin with pytest test classes.
    """
    
    def setup_database_test(self, db_connection: DatabaseConnection):
        """Initialize database test setup."""
        self.db = db_connection
        self.uow = SQLiteUnitOfWork(db_connection)
        self.training_service = TrainingTransactionService(db_connection)
    
    def create_test_robot(self, robot_id: str = "test-robot"):
        """Create a test robot for database operations."""
        from src.genesis_humanoid_rl.domain.model.aggregates import HumanoidRobot, RobotType
        from src.genesis_humanoid_rl.domain.model.value_objects import RobotId
        
        return HumanoidRobot(
            robot_id=RobotId.from_string(robot_id),
            robot_type=RobotType.UNITREE_G1,
            name=f"Test Robot {robot_id}",
            joint_count=35,
            height=1.2,
            weight=35.0
        )
    
    def create_test_session(self, session_id: str = "test-session", robot_id: str = "test-robot"):
        """Create a test learning session."""
        from src.genesis_humanoid_rl.domain.model.aggregates import LearningSession
        from src.genesis_humanoid_rl.domain.model.value_objects import SessionId, RobotId, PlanId
        from datetime import datetime
        
        return LearningSession(
            session_id=SessionId.from_string(session_id),
            robot_id=RobotId.from_string(robot_id),
            plan_id=PlanId.from_string(f"plan-{session_id}"),
            created_at=datetime.now()
        )


@pytest.fixture
def database_test_case(database_connection_fixture):
    """Pytest fixture for database test case helper."""
    test_case = DatabaseTestCase()
    test_case.setup_database_test(database_connection_fixture)
    return test_case


# Performance testing fixtures
@pytest.fixture
def performance_database(temp_database_path):
    """Database fixture optimized for performance testing."""
    with database_connection(temp_database_path) as db:
        # Configure for performance testing
        db.execute("PRAGMA synchronous = OFF")
        db.execute("PRAGMA journal_mode = MEMORY")
        db.execute("PRAGMA cache_size = 10000")
        yield db


@pytest.fixture
def isolated_transaction(unit_of_work_fixture):
    """
    Fixture that provides isolated transaction for each test.
    
    Automatically rolls back transaction after test to ensure isolation.
    """
    uow = unit_of_work_fixture
    
    # Start transaction
    uow.begin()
    
    yield uow
    
    # Always rollback to ensure test isolation
    try:
        uow.rollback()
    except Exception:
        pass  # Ignore rollback errors


# Mock fixtures for external dependencies
@pytest.fixture
def mock_database_connection():
    """Mock database connection for unit tests that don't need real database."""
    mock_db = Mock(spec=DatabaseConnection)
    mock_db.execute.return_value = None
    mock_db.fetch_one.return_value = None
    mock_db.fetch_all.return_value = []
    mock_db.close.return_value = None
    return mock_db


@pytest.fixture
def mock_unit_of_work(mock_database_connection):
    """Mock unit of work for testing business logic without database."""
    mock_uow = Mock(spec=SQLiteUnitOfWork)
    mock_uow.robots = Mock()
    mock_uow.sessions = Mock()
    mock_uow.episodes = Mock()
    mock_uow.commit.return_value = None
    mock_uow.rollback.return_value = None
    return mock_uow