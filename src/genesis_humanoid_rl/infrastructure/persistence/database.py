"""Database connection and session management."""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Manages SQLite database connections and sessions."""

    def __init__(self, db_path: str = "genesis_humanoid_rl.db"):
        """Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._connection: Optional[sqlite3.Connection] = None
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize database schema if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    id TEXT PRIMARY KEY,
                    robot_id TEXT NOT NULL,
                    curriculum_plan_id TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    total_reward REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learning_episodes (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    motion_command TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    total_reward REAL DEFAULT 0.0,
                    success BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES learning_sessions(id)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS humanoid_robots (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    configuration TEXT NOT NULL,
                    learned_skills TEXT NOT NULL,
                    skill_history TEXT NOT NULL,
                    performance_history TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS curriculum_plans (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    stages TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_stage_index INTEGER DEFAULT 0,
                    difficulty_params TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS domain_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    aggregate_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    occurred_at REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_episodes_session ON learning_episodes(session_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_aggregate ON domain_events(aggregate_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_type ON domain_events(event_type)"
            )

            conn.commit()

        logger.info(f"Database initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit mode
            )
            self._connection.row_factory = sqlite3.Row

            # Enable foreign keys
            self._connection.execute("PRAGMA foreign_keys = ON")

            # Performance optimizations
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = NORMAL")

        return self._connection

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a database transaction context.

        Yields:
            Database connection in transaction mode
        """
        conn = self._get_connection()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def execute(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> sqlite3.Cursor:
        """Execute a database query.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Cursor with query results
        """
        conn = self._get_connection()
        if params:
            return conn.execute(query, params)
        return conn.execute(query)

    def fetch_one(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[sqlite3.Row]:
        """Execute query and fetch one result.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            Single row result or None
        """
        cursor = self.execute(query, params)
        return cursor.fetchone()

    def fetch_all(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> list[sqlite3.Row]:
        """Execute query and fetch all results.

        Args:
            query: SQL query to execute
            params: Query parameters

        Returns:
            List of row results
        """
        cursor = self.execute(query, params)
        return cursor.fetchall()

    def close(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
