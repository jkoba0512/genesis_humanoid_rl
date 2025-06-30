"""
Database migration management system.

Provides versioned schema migrations with data preservation and rollback capabilities.
"""

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import json

from .schema_design import OptimizedSchemaDesign
from .database import DatabaseConnection

logger = logging.getLogger(__name__)


@dataclass
class MigrationRecord:
    """Record of a completed migration."""

    version: str
    name: str
    applied_at: datetime
    execution_time_seconds: float
    rollback_sql: Optional[str] = None


class MigrationError(Exception):
    """Exception raised during migration operations."""

    def __init__(
        self,
        message: str,
        migration_version: str = None,
        original_error: Exception = None,
    ):
        super().__init__(message)
        self.migration_version = migration_version
        self.original_error = original_error


class DatabaseMigrationManager:
    """
    Manages database schema migrations with versioning and rollback support.

    Features:
    - Version-controlled schema changes
    - Data preservation during migrations
    - Rollback capabilities for failed migrations
    - Migration status tracking
    - Performance monitoring
    """

    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        self.schema_design = OptimizedSchemaDesign()
        self._ensure_migration_table()

    def _ensure_migration_table(self) -> None:
        """Create migration tracking table if it doesn't exist."""
        self.db_connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL,
                execution_time_seconds REAL NOT NULL,
                rollback_sql TEXT,
                checksum TEXT NOT NULL
            )
        """
        )

    def get_current_version(self) -> Optional[str]:
        """Get the current schema version."""
        result = self.db_connection.fetch_one(
            "SELECT version FROM schema_migrations ORDER BY applied_at DESC LIMIT 1"
        )
        return result["version"] if result else None

    def get_migration_history(self) -> List[MigrationRecord]:
        """Get complete migration history."""
        results = self.db_connection.fetch_all(
            "SELECT * FROM schema_migrations ORDER BY applied_at ASC"
        )

        return [
            MigrationRecord(
                version=row["version"],
                name=row["name"],
                applied_at=datetime.fromisoformat(row["applied_at"]),
                execution_time_seconds=row["execution_time_seconds"],
                rollback_sql=row["rollback_sql"],
            )
            for row in results
        ]

    def is_migration_applied(self, version: str) -> bool:
        """Check if a migration version has been applied."""
        result = self.db_connection.fetch_one(
            "SELECT 1 FROM schema_migrations WHERE version = :version",
            {"version": version},
        )
        return result is not None

    @contextmanager
    def migration_transaction(
        self, version: str, name: str
    ) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for migration transactions with automatic rollback on failure."""
        start_time = datetime.now()

        with self.db_connection.transaction() as conn:
            try:
                yield conn

                # Record successful migration
                execution_time = (datetime.now() - start_time).total_seconds()
                self._record_migration(version, name, execution_time)

                logger.info(
                    f"Migration {version} '{name}' completed successfully in {execution_time:.2f}s"
                )

            except Exception as e:
                logger.error(f"Migration {version} '{name}' failed: {str(e)}")
                raise MigrationError(
                    f"Migration {version} failed: {str(e)}",
                    migration_version=version,
                    original_error=e,
                )

    def migrate_to_optimized_schema(self) -> None:
        """Migrate to the optimized normalized schema."""
        version = "v2.0.0"
        name = "Optimized Normalized Schema"

        if self.is_migration_applied(version):
            logger.info(f"Migration {version} already applied")
            return

        logger.info(f"Starting migration to {version}: {name}")

        with self.migration_transaction(version, name) as conn:
            # Disable foreign keys for entire migration
            conn.execute("PRAGMA foreign_keys = OFF")

            try:
                # Step 1: Backup existing data
                backed_up_data = self._backup_existing_data(conn)

                # Step 2: Create new schema
                self._create_optimized_schema(conn)

                # Step 3: Migrate data to new schema
                self._migrate_data_to_new_schema(conn, backed_up_data)

                # Step 4: Validate migration
                self._validate_migration(conn, backed_up_data)

                logger.info(f"Successfully migrated to optimized schema {version}")

            finally:
                # Re-enable foreign keys
                conn.execute("PRAGMA foreign_keys = ON")

    def _backup_existing_data(
        self, conn: sqlite3.Connection
    ) -> Dict[str, List[sqlite3.Row]]:
        """Backup all existing data before migration."""
        logger.info("Backing up existing data...")

        backup_data = {}

        # Get list of existing tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        existing_tables = [row[0] for row in cursor.fetchall()]

        # Backup data from each table
        for table_name in existing_tables:
            if table_name != "schema_migrations":
                cursor = conn.execute(f"SELECT * FROM {table_name}")
                backup_data[table_name] = cursor.fetchall()
                logger.debug(
                    f"Backed up {len(backup_data[table_name])} rows from {table_name}"
                )

        logger.info(f"Backed up data from {len(backup_data)} tables")
        return backup_data

    def _create_optimized_schema(self, conn: sqlite3.Connection) -> None:
        """Create the new optimized schema."""
        logger.info("Creating optimized schema...")

        # First, drop existing tables that conflict (but preserve data - it's already backed up)
        legacy_tables = [
            "learning_episodes",  # Drop in order to avoid FK constraints
            "learning_sessions",
            "curriculum_plans",
            "humanoid_robots",
            "domain_events",
        ]

        for table_name in legacy_tables:
            try:
                conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                logger.debug(f"Dropped legacy table {table_name}")
            except Exception as e:
                logger.warning(f"Could not drop table {table_name}: {e}")

        # Create new optimized schema
        migration_statements = self.schema_design.get_migration_sql()

        for i, statement in enumerate(migration_statements):
            try:
                conn.execute(statement)
                logger.debug(
                    f"Executed migration statement {i+1}/{len(migration_statements)}"
                )
            except Exception as e:
                logger.error(f"Failed to execute statement {i+1}: {statement}")
                raise

        logger.info(
            f"Created optimized schema with {len(migration_statements)} statements"
        )

    def _migrate_data_to_new_schema(
        self, conn: sqlite3.Connection, backup_data: Dict[str, List[sqlite3.Row]]
    ) -> None:
        """Migrate backed up data to the new schema structure."""
        logger.info("Migrating data to new schema...")

        # Migrate robots data
        if "humanoid_robots" in backup_data:
            self._migrate_robots_data(conn, backup_data["humanoid_robots"])

        # Migrate curriculum plans
        if "curriculum_plans" in backup_data:
            self._migrate_curriculum_plans_data(conn, backup_data["curriculum_plans"])

        # Migrate learning sessions
        if "learning_sessions" in backup_data:
            self._migrate_learning_sessions_data(conn, backup_data["learning_sessions"])

        # Migrate learning episodes
        if "learning_episodes" in backup_data:
            self._migrate_learning_episodes_data(conn, backup_data["learning_episodes"])

        # Migrate domain events
        if "domain_events" in backup_data:
            self._migrate_domain_events_data(conn, backup_data["domain_events"])

        logger.info("Data migration completed")

    def _migrate_robots_data(
        self, conn: sqlite3.Connection, robots_data: List[sqlite3.Row]
    ) -> None:
        """Migrate robots data to normalized structure."""
        logger.debug("Migrating robots data...")

        for robot_row in robots_data:
            # Parse stored JSON data
            learned_skills = (
                json.loads(robot_row["learned_skills"])
                if robot_row["learned_skills"]
                else {}
            )

            # Insert into normalized robots table
            conn.execute(
                """
                INSERT INTO robots (id, name, robot_type, joint_count, height, weight, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    robot_row["id"],
                    robot_row["name"],
                    "UNITREE_G1",  # Default robot type
                    35,  # Default joint count for G1
                    1.2,  # Default height
                    35.0,  # Default weight
                    robot_row["created_at"],
                    robot_row["updated_at"],
                ),
            )

            # Insert learned skills into robot_skills table
            for skill_type, skill_data in learned_skills.items():
                if isinstance(skill_data, dict):
                    # First ensure skill definition exists
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO skill_definitions 
                        (skill_type, name, description, category, difficulty_level)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            skill_type,
                            skill_type.replace(
                                "_", " "
                            ).title(),  # Convert FORWARD_WALKING to Forward Walking
                            f"Migrated skill: {skill_type}",
                            "locomotion",  # Default category
                            1.0,  # Default difficulty
                        ),
                    )

                    proficiency = (
                        skill_data.get("proficiency_score", 0.0)
                        if isinstance(skill_data, dict)
                        else 0.0
                    )
                    mastery_level = (
                        skill_data.get("mastery_level", "BEGINNER")
                        if isinstance(skill_data, dict)
                        else "BEGINNER"
                    )

                    conn.execute(
                        """
                        INSERT OR IGNORE INTO robot_skills 
                        (robot_id, skill_type, mastery_level, proficiency_score, acquired_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        (
                            robot_row["id"],
                            skill_type,
                            mastery_level,
                            proficiency,
                            robot_row["created_at"],
                            robot_row["updated_at"],
                        ),
                    )

        logger.debug(f"Migrated {len(robots_data)} robots")

    def _migrate_curriculum_plans_data(
        self, conn: sqlite3.Connection, plans_data: List[sqlite3.Row]
    ) -> None:
        """Migrate curriculum plans to normalized structure."""
        logger.debug("Migrating curriculum plans data...")

        for plan_row in plans_data:
            # Insert main plan record
            conn.execute(
                """
                INSERT INTO curriculum_plans (id, name, description, status, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    plan_row["id"],
                    plan_row["name"],
                    f"Migrated plan: {plan_row['name']}",
                    plan_row["status"],
                    "1.0",  # Default version
                    plan_row["created_at"],
                    plan_row["updated_at"],
                ),
            )

            # Parse and migrate stages
            stages_data = json.loads(plan_row["stages"]) if plan_row["stages"] else []
            for i, stage_data in enumerate(stages_data):
                if isinstance(stage_data, dict):
                    conn.execute(
                        """
                        INSERT INTO curriculum_stages 
                        (plan_id, stage_order, stage_id, name, stage_type, difficulty_level, 
                         target_success_rate, min_episodes, expected_duration_episodes, 
                         advancement_criteria, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            plan_row["id"],
                            i + 1,
                            stage_data.get("id", f"stage_{i}"),
                            stage_data.get("name", f"Stage {i+1}"),
                            "FOUNDATION",  # Default stage type
                            stage_data.get("difficulty_level", 1.0),
                            stage_data.get("target_success_rate", 0.7),
                            stage_data.get("min_episodes", 10),
                            stage_data.get("expected_duration_episodes", 50),
                            json.dumps(stage_data.get("advancement_criteria", {})),
                            plan_row["created_at"],
                        ),
                    )

        logger.debug(f"Migrated {len(plans_data)} curriculum plans")

    def _migrate_learning_sessions_data(
        self, conn: sqlite3.Connection, sessions_data: List[sqlite3.Row]
    ) -> None:
        """Migrate learning sessions data."""
        logger.debug("Migrating learning sessions data...")

        for session_row in sessions_data:
            conn.execute(
                """
                INSERT INTO learning_sessions 
                (id, robot_id, curriculum_plan_id, session_name, status, start_time, end_time,
                 total_episodes, successful_episodes, total_reward, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_row["id"],
                    session_row["robot_id"],
                    session_row["curriculum_plan_id"],
                    f"Migrated Session {session_row['id'][:8]}",
                    "completed",  # Default status
                    session_row["start_time"],
                    session_row["end_time"],
                    0,  # Will be calculated from episodes
                    0,  # Will be calculated from episodes
                    session_row["total_reward"],
                    session_row["created_at"],
                    session_row["updated_at"],
                ),
            )

        logger.debug(f"Migrated {len(sessions_data)} learning sessions")

    def _migrate_learning_episodes_data(
        self, conn: sqlite3.Connection, episodes_data: List[sqlite3.Row]
    ) -> None:
        """Migrate learning episodes data."""
        logger.debug("Migrating learning episodes data...")

        for episode_row in episodes_data:
            # Determine episode outcome from success field
            success = (
                episode_row["success"] if "success" in episode_row.keys() else False
            )
            outcome = "success" if success else "failure"

            conn.execute(
                """
                INSERT INTO learning_episodes 
                (id, session_id, episode_number, status, outcome, start_time, end_time,
                 step_count, total_reward, average_reward, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    episode_row["id"],
                    episode_row["session_id"],
                    1,  # Default episode number (could be calculated)
                    "completed",
                    outcome,
                    episode_row["start_time"],
                    episode_row["end_time"],
                    1000,  # Default step count
                    episode_row["total_reward"],
                    episode_row["total_reward"] / 1000,  # Approximate average
                    episode_row["created_at"],
                ),
            )

        logger.debug(f"Migrated {len(episodes_data)} learning episodes")

    def _migrate_domain_events_data(
        self, conn: sqlite3.Connection, events_data: List[sqlite3.Row]
    ) -> None:
        """Migrate domain events data."""
        logger.debug("Migrating domain events data...")

        for event_row in events_data:
            conn.execute(
                """
                INSERT INTO domain_events 
                (stream_id, event_type, event_version, event_data, metadata, occurred_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event_row["aggregate_id"],  # Map aggregate_id to stream_id
                    event_row["event_type"],
                    1,  # Default version
                    event_row["payload"],  # Map payload to event_data
                    "{}",  # Default empty metadata
                    event_row["occurred_at"],
                    event_row["created_at"],
                ),
            )

        logger.debug(f"Migrated {len(events_data)} domain events")

    def _validate_migration(
        self, conn: sqlite3.Connection, original_data: Dict[str, List[sqlite3.Row]]
    ) -> None:
        """Validate that migration preserved data integrity."""
        logger.info("Validating migration...")

        validations = []

        # Validate robots count
        if "humanoid_robots" in original_data:
            original_count = len(original_data["humanoid_robots"])
            cursor = conn.execute("SELECT COUNT(*) FROM robots")
            new_count = cursor.fetchone()[0]
            validations.append(("robots", original_count, new_count))

        # Validate sessions count
        if "learning_sessions" in original_data:
            original_count = len(original_data["learning_sessions"])
            cursor = conn.execute("SELECT COUNT(*) FROM learning_sessions")
            new_count = cursor.fetchone()[0]
            validations.append(("learning_sessions", original_count, new_count))

        # Validate episodes count
        if "learning_episodes" in original_data:
            original_count = len(original_data["learning_episodes"])
            cursor = conn.execute("SELECT COUNT(*) FROM learning_episodes")
            new_count = cursor.fetchone()[0]
            validations.append(("learning_episodes", original_count, new_count))

        # Check validation results
        for table_name, original_count, new_count in validations:
            if original_count != new_count:
                raise MigrationError(
                    f"Data validation failed for {table_name}: "
                    f"original={original_count}, migrated={new_count}"
                )
            logger.debug(f"Validation passed for {table_name}: {new_count} records")

        logger.info(f"Migration validation passed for {len(validations)} tables")

    def _record_migration(self, version: str, name: str, execution_time: float) -> None:
        """Record successful migration in migration history."""
        checksum = self._calculate_schema_checksum()

        self.db_connection.execute(
            """
            INSERT INTO schema_migrations 
            (version, name, applied_at, execution_time_seconds, checksum)
            VALUES (:version, :name, :applied_at, :execution_time_seconds, :checksum)
        """,
            {
                "version": version,
                "name": name,
                "applied_at": datetime.now().isoformat(),
                "execution_time_seconds": execution_time,
                "checksum": checksum,
            },
        )

    def _calculate_schema_checksum(self) -> str:
        """Calculate checksum of current schema for validation."""
        # Get all table schemas
        result = self.db_connection.fetch_all(
            "SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name"
        )

        schema_sql = "".join(row["sql"] or "" for row in result)

        # Simple hash for schema validation
        import hashlib

        return hashlib.md5(schema_sql.encode()).hexdigest()

    def get_migration_status(self) -> Dict[str, any]:
        """Get comprehensive migration status information."""
        current_version = self.get_current_version()
        migration_history = self.get_migration_history()

        # Analyze schema
        cursor = self.db_connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        table_names = [row[0] for row in cursor.fetchall()]

        # Check if optimized schema is applied
        optimized_tables = set(self.schema_design.tables.keys())
        current_tables = set(table_names) - {"schema_migrations"}

        has_optimized_schema = optimized_tables.issubset(current_tables)

        return {
            "current_version": current_version,
            "has_optimized_schema": has_optimized_schema,
            "total_migrations": len(migration_history),
            "current_tables": sorted(current_tables),
            "optimized_tables": sorted(optimized_tables),
            "missing_tables": sorted(optimized_tables - current_tables),
            "migration_history": [
                {
                    "version": m.version,
                    "name": m.name,
                    "applied_at": m.applied_at.isoformat(),
                    "execution_time": m.execution_time_seconds,
                }
                for m in migration_history
            ],
        }
