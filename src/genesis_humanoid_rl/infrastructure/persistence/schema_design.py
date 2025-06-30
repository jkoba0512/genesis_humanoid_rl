"""
Optimized database schema design for performance and normalization.

This module provides a comprehensive, normalized database schema design
optimized for humanoid robotics learning data with performance considerations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TableOptimization(Enum):
    """Database optimization strategies."""

    # Indexing strategies
    BTREE_INDEX = "btree"
    HASH_INDEX = "hash"
    COMPOSITE_INDEX = "composite"
    PARTIAL_INDEX = "partial"

    # Storage optimizations
    COMPRESSED_STORAGE = "compressed"
    PARTITIONED_TABLE = "partitioned"
    MATERIALIZED_VIEW = "materialized_view"

    # Query optimizations
    QUERY_CACHE = "query_cache"
    CONNECTION_POOLING = "connection_pooling"
    PREPARED_STATEMENTS = "prepared_statements"


@dataclass
class IndexDefinition:
    """Definition of a database index for performance optimization."""

    name: str
    table: str
    columns: List[str]
    unique: bool = False
    partial_condition: Optional[str] = None
    optimization_type: TableOptimization = TableOptimization.BTREE_INDEX
    estimated_cardinality: Optional[int] = None

    def to_sql(self) -> str:
        """Generate SQL DDL for this index."""
        unique_clause = "UNIQUE " if self.unique else ""
        columns_clause = ", ".join(self.columns)
        partial_clause = (
            f" WHERE {self.partial_condition}" if self.partial_condition else ""
        )

        return f"CREATE {unique_clause}INDEX IF NOT EXISTS {self.name} ON {self.table}({columns_clause}){partial_clause}"


@dataclass
class TableDefinition:
    """Definition of a database table with optimization metadata."""

    name: str
    columns: Dict[str, str]  # column_name -> column_type
    primary_key: List[str]
    foreign_keys: Dict[str, str] = None  # column -> referenced_table.column
    indexes: List[IndexDefinition] = None
    constraints: List[str] = None
    estimated_rows: Optional[int] = None

    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = {}
        if self.indexes is None:
            self.indexes = []
        if self.constraints is None:
            self.constraints = []


class OptimizedSchemaDesign:
    """
    Comprehensive database schema design optimized for performance.

    Features:
    - Proper normalization (3NF) to eliminate redundancy
    - Strategic indexing for common query patterns
    - Partitioning strategies for large datasets
    - Performance-optimized data types
    - Comprehensive foreign key relationships
    """

    def __init__(self):
        self.tables: Dict[str, TableDefinition] = {}
        self.views: Dict[str, str] = {}
        self.triggers: Dict[str, str] = {}
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize the complete optimized schema."""
        self._create_core_entity_tables()
        self._create_skill_and_performance_tables()
        self._create_training_data_tables()
        self._create_configuration_tables()
        self._create_event_and_audit_tables()
        self._create_performance_views()
        self._create_audit_triggers()

    def _create_core_entity_tables(self):
        """Create normalized tables for core domain entities."""

        # Robots table (normalized)
        self.tables["robots"] = TableDefinition(
            name="robots",
            columns={
                "id": "TEXT PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "robot_type": "TEXT NOT NULL",
                "joint_count": "INTEGER NOT NULL",
                "height": "REAL NOT NULL",
                "weight": "REAL NOT NULL",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "status": "TEXT DEFAULT 'active'",
                "metadata": "TEXT",  # JSON for non-critical configuration
            },
            primary_key=["id"],
            indexes=[
                IndexDefinition("idx_robots_type", "robots", ["robot_type"]),
                IndexDefinition("idx_robots_status", "robots", ["status"]),
                IndexDefinition("idx_robots_created", "robots", ["created_at"]),
            ],
            constraints=[
                "CHECK (joint_count > 0)",
                "CHECK (height > 0)",
                "CHECK (weight > 0)",
                "CHECK (status IN ('active', 'inactive', 'maintenance'))",
            ],
            estimated_rows=1000,
        )

        # Learning sessions (optimized for queries)
        self.tables["learning_sessions"] = TableDefinition(
            name="learning_sessions",
            columns={
                "id": "TEXT PRIMARY KEY",
                "robot_id": "TEXT NOT NULL",
                "curriculum_plan_id": "TEXT NOT NULL",
                "session_name": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'created'",
                "start_time": "TIMESTAMP",
                "end_time": "TIMESTAMP",
                "target_duration_hours": "REAL",
                "max_episodes": "INTEGER DEFAULT 1000",
                "total_episodes": "INTEGER DEFAULT 0",
                "successful_episodes": "INTEGER DEFAULT 0",
                "total_reward": "REAL DEFAULT 0.0",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            foreign_keys={
                "robot_id": "robots.id",
                "curriculum_plan_id": "curriculum_plans.id",
            },
            indexes=[
                IndexDefinition(
                    "idx_sessions_robot", "learning_sessions", ["robot_id"]
                ),
                IndexDefinition(
                    "idx_sessions_plan", "learning_sessions", ["curriculum_plan_id"]
                ),
                IndexDefinition("idx_sessions_status", "learning_sessions", ["status"]),
                IndexDefinition(
                    "idx_sessions_active",
                    "learning_sessions",
                    ["robot_id", "status"],
                    partial_condition="status = 'active'",
                ),
                IndexDefinition(
                    "idx_sessions_time_range",
                    "learning_sessions",
                    ["start_time", "end_time"],
                ),
            ],
            estimated_rows=10000,
        )

        # Learning episodes (partitioned by session for performance)
        self.tables["learning_episodes"] = TableDefinition(
            name="learning_episodes",
            columns={
                "id": "TEXT PRIMARY KEY",
                "session_id": "TEXT NOT NULL",
                "episode_number": "INTEGER NOT NULL",
                "target_skill_type": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'pending'",
                "outcome": "TEXT",
                "start_time": "TIMESTAMP NOT NULL",
                "end_time": "TIMESTAMP",
                "step_count": "INTEGER DEFAULT 0",
                "max_steps": "INTEGER DEFAULT 1000",
                "total_reward": "REAL DEFAULT 0.0",
                "average_reward": "REAL DEFAULT 0.0",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            foreign_keys={"session_id": "learning_sessions.id"},
            indexes=[
                IndexDefinition(
                    "idx_episodes_session", "learning_episodes", ["session_id"]
                ),
                IndexDefinition(
                    "idx_episodes_session_number",
                    "learning_episodes",
                    ["session_id", "episode_number"],
                    unique=True,
                ),
                IndexDefinition(
                    "idx_episodes_skill", "learning_episodes", ["target_skill_type"]
                ),
                IndexDefinition(
                    "idx_episodes_outcome", "learning_episodes", ["outcome"]
                ),
                IndexDefinition(
                    "idx_episodes_performance",
                    "learning_episodes",
                    ["total_reward", "step_count"],
                ),
                IndexDefinition(
                    "idx_episodes_time", "learning_episodes", ["start_time"]
                ),
            ],
            estimated_rows=1000000,
        )

        # Curriculum plans (normalized)
        self.tables["curriculum_plans"] = TableDefinition(
            name="curriculum_plans",
            columns={
                "id": "TEXT PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "description": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'draft'",
                "version": "TEXT NOT NULL",
                "created_by": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            indexes=[
                IndexDefinition("idx_plans_status", "curriculum_plans", ["status"]),
                IndexDefinition(
                    "idx_plans_version",
                    "curriculum_plans",
                    ["name", "version"],
                    unique=True,
                ),
            ],
            estimated_rows=100,
        )

    def _create_skill_and_performance_tables(self):
        """Create normalized tables for skills and performance tracking."""

        # Skill definitions (master data)
        self.tables["skill_definitions"] = TableDefinition(
            name="skill_definitions",
            columns={
                "skill_type": "TEXT PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "description": "TEXT",
                "category": "TEXT NOT NULL",
                "difficulty_level": "REAL NOT NULL DEFAULT 1.0",
                "prerequisite_skills": "TEXT",  # JSON array
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["skill_type"],
            indexes=[
                IndexDefinition(
                    "idx_skills_category", "skill_definitions", ["category"]
                ),
                IndexDefinition(
                    "idx_skills_difficulty", "skill_definitions", ["difficulty_level"]
                ),
            ],
            estimated_rows=50,
        )

        # Robot skills (many-to-many with proficiency tracking)
        self.tables["robot_skills"] = TableDefinition(
            name="robot_skills",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "robot_id": "TEXT NOT NULL",
                "skill_type": "TEXT NOT NULL",
                "mastery_level": "TEXT NOT NULL",
                "proficiency_score": "REAL NOT NULL DEFAULT 0.0",
                "confidence_score": "REAL DEFAULT 0.0",
                "last_practiced": "TIMESTAMP",
                "practice_count": "INTEGER DEFAULT 0",
                "acquired_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            foreign_keys={
                "robot_id": "robots.id",
                "skill_type": "skill_definitions.skill_type",
            },
            indexes=[
                IndexDefinition("idx_robot_skills_robot", "robot_skills", ["robot_id"]),
                IndexDefinition(
                    "idx_robot_skills_skill", "robot_skills", ["skill_type"]
                ),
                IndexDefinition(
                    "idx_robot_skills_unique",
                    "robot_skills",
                    ["robot_id", "skill_type"],
                    unique=True,
                ),
                IndexDefinition(
                    "idx_robot_skills_proficiency",
                    "robot_skills",
                    ["proficiency_score"],
                ),
                IndexDefinition(
                    "idx_robot_skills_mastery", "robot_skills", ["mastery_level"]
                ),
            ],
            estimated_rows=50000,
        )

        # Performance metrics (time-series data)
        self.tables["performance_metrics"] = TableDefinition(
            name="performance_metrics",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "episode_id": "TEXT NOT NULL",
                "metric_type": "TEXT NOT NULL",
                "metric_name": "TEXT NOT NULL",
                "value": "REAL NOT NULL",
                "timestamp": "TIMESTAMP NOT NULL",
                "step_number": "INTEGER",
                "metadata": "TEXT",  # JSON for additional context
            },
            primary_key=["id"],
            foreign_keys={"episode_id": "learning_episodes.id"},
            indexes=[
                IndexDefinition(
                    "idx_metrics_episode", "performance_metrics", ["episode_id"]
                ),
                IndexDefinition(
                    "idx_metrics_type",
                    "performance_metrics",
                    ["metric_type", "metric_name"],
                ),
                IndexDefinition(
                    "idx_metrics_time", "performance_metrics", ["timestamp"]
                ),
                IndexDefinition(
                    "idx_metrics_composite",
                    "performance_metrics",
                    ["episode_id", "metric_type", "step_number"],
                ),
            ],
            estimated_rows=10000000,
        )

        # Skill assessments (detailed evaluation records)
        self.tables["skill_assessments"] = TableDefinition(
            name="skill_assessments",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "robot_id": "TEXT NOT NULL",
                "skill_type": "TEXT NOT NULL",
                "episode_id": "TEXT",
                "assessment_score": "REAL NOT NULL",
                "confidence": "REAL NOT NULL",
                "evidence_quality": "REAL NOT NULL",
                "assessor_type": "TEXT NOT NULL",  # 'automated', 'manual', 'hybrid'
                "assessment_data": "TEXT",  # JSON for detailed results
                "assessed_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            foreign_keys={
                "robot_id": "robots.id",
                "skill_type": "skill_definitions.skill_type",
                "episode_id": "learning_episodes.id",
            },
            indexes=[
                IndexDefinition(
                    "idx_assessments_robot", "skill_assessments", ["robot_id"]
                ),
                IndexDefinition(
                    "idx_assessments_skill", "skill_assessments", ["skill_type"]
                ),
                IndexDefinition(
                    "idx_assessments_score", "skill_assessments", ["assessment_score"]
                ),
                IndexDefinition(
                    "idx_assessments_time", "skill_assessments", ["assessed_at"]
                ),
            ],
            estimated_rows=500000,
        )

    def _create_training_data_tables(self):
        """Create tables for training data and motion tracking."""

        # Motion commands (normalized from episodes)
        self.tables["motion_commands"] = TableDefinition(
            name="motion_commands",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "episode_id": "TEXT NOT NULL",
                "command_sequence": "INTEGER NOT NULL",
                "motion_type": "TEXT NOT NULL",
                "velocity": "REAL NOT NULL",
                "duration": "REAL",
                "complexity_score": "REAL",
                "parameters": "TEXT",  # JSON for additional parameters
                "executed_at": "TIMESTAMP NOT NULL",
            },
            primary_key=["id"],
            foreign_keys={"episode_id": "learning_episodes.id"},
            indexes=[
                IndexDefinition(
                    "idx_commands_episode", "motion_commands", ["episode_id"]
                ),
                IndexDefinition(
                    "idx_commands_sequence",
                    "motion_commands",
                    ["episode_id", "command_sequence"],
                    unique=True,
                ),
                IndexDefinition(
                    "idx_commands_type", "motion_commands", ["motion_type"]
                ),
                IndexDefinition(
                    "idx_commands_time", "motion_commands", ["executed_at"]
                ),
            ],
            estimated_rows=5000000,
        )

        # Movement trajectories (spatial-temporal data)
        self.tables["movement_trajectories"] = TableDefinition(
            name="movement_trajectories",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "episode_id": "TEXT NOT NULL",
                "step_number": "INTEGER NOT NULL",
                "position_x": "REAL NOT NULL",
                "position_y": "REAL NOT NULL",
                "position_z": "REAL NOT NULL",
                "orientation_x": "REAL NOT NULL",
                "orientation_y": "REAL NOT NULL",
                "orientation_z": "REAL NOT NULL",
                "orientation_w": "REAL NOT NULL",
                "timestamp": "TIMESTAMP NOT NULL",
                "velocity_magnitude": "REAL",
                "acceleration_magnitude": "REAL",
            },
            primary_key=["id"],
            foreign_keys={"episode_id": "learning_episodes.id"},
            indexes=[
                IndexDefinition(
                    "idx_trajectories_episode", "movement_trajectories", ["episode_id"]
                ),
                IndexDefinition(
                    "idx_trajectories_step",
                    "movement_trajectories",
                    ["episode_id", "step_number"],
                    unique=True,
                ),
                IndexDefinition(
                    "idx_trajectories_time", "movement_trajectories", ["timestamp"]
                ),
                IndexDefinition(
                    "idx_trajectories_spatial",
                    "movement_trajectories",
                    ["position_x", "position_y", "position_z"],
                ),
            ],
            estimated_rows=50000000,
        )

    def _create_configuration_tables(self):
        """Create tables for configuration management."""

        # Curriculum stages (normalized from plans)
        self.tables["curriculum_stages"] = TableDefinition(
            name="curriculum_stages",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "plan_id": "TEXT NOT NULL",
                "stage_order": "INTEGER NOT NULL",
                "stage_id": "TEXT NOT NULL",
                "name": "TEXT NOT NULL",
                "stage_type": "TEXT NOT NULL",
                "difficulty_level": "REAL NOT NULL DEFAULT 1.0",
                "target_success_rate": "REAL NOT NULL DEFAULT 0.7",
                "min_episodes": "INTEGER NOT NULL DEFAULT 10",
                "expected_duration_episodes": "INTEGER NOT NULL DEFAULT 50",
                "advancement_criteria": "TEXT NOT NULL",  # JSON
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            foreign_keys={"plan_id": "curriculum_plans.id"},
            indexes=[
                IndexDefinition("idx_stages_plan", "curriculum_stages", ["plan_id"]),
                IndexDefinition(
                    "idx_stages_order",
                    "curriculum_stages",
                    ["plan_id", "stage_order"],
                    unique=True,
                ),
                IndexDefinition(
                    "idx_stages_difficulty", "curriculum_stages", ["difficulty_level"]
                ),
            ],
            estimated_rows=1000,
        )

        # Stage skills (many-to-many)
        self.tables["stage_skills"] = TableDefinition(
            name="stage_skills",
            columns={
                "stage_id": "INTEGER NOT NULL",
                "skill_type": "TEXT NOT NULL",
                "skill_role": "TEXT NOT NULL",  # 'target', 'prerequisite', 'supporting'
                "weight": "REAL DEFAULT 1.0",
            },
            primary_key=["stage_id", "skill_type"],
            foreign_keys={
                "stage_id": "curriculum_stages.id",
                "skill_type": "skill_definitions.skill_type",
            },
            indexes=[
                IndexDefinition("idx_stage_skills_stage", "stage_skills", ["stage_id"]),
                IndexDefinition(
                    "idx_stage_skills_skill", "stage_skills", ["skill_type"]
                ),
                IndexDefinition(
                    "idx_stage_skills_role", "stage_skills", ["skill_role"]
                ),
            ],
            estimated_rows=5000,
        )

    def _create_event_and_audit_tables(self):
        """Create tables for event sourcing and audit trails."""

        # Domain events (optimized for event sourcing)
        self.tables["domain_events"] = TableDefinition(
            name="domain_events",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "stream_id": "TEXT NOT NULL",  # Aggregate ID
                "event_type": "TEXT NOT NULL",
                "event_version": "INTEGER NOT NULL DEFAULT 1",
                "event_data": "TEXT NOT NULL",  # JSON payload
                "metadata": "TEXT",  # JSON metadata
                "occurred_at": "TIMESTAMP NOT NULL",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
            },
            primary_key=["id"],
            indexes=[
                IndexDefinition("idx_events_stream", "domain_events", ["stream_id"]),
                IndexDefinition("idx_events_type", "domain_events", ["event_type"]),
                IndexDefinition("idx_events_time", "domain_events", ["occurred_at"]),
                IndexDefinition(
                    "idx_events_stream_order",
                    "domain_events",
                    ["stream_id", "created_at"],
                ),
                # Note: Removed datetime-based partial index due to SQLite non-deterministic function restriction
                IndexDefinition("idx_events_created", "domain_events", ["created_at"]),
            ],
            estimated_rows=5000000,
        )

        # Audit log (for compliance and debugging)
        self.tables["audit_log"] = TableDefinition(
            name="audit_log",
            columns={
                "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
                "table_name": "TEXT NOT NULL",
                "record_id": "TEXT NOT NULL",
                "operation": "TEXT NOT NULL",  # 'INSERT', 'UPDATE', 'DELETE'
                "old_values": "TEXT",  # JSON
                "new_values": "TEXT",  # JSON
                "changed_by": "TEXT",
                "changed_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "session_id": "TEXT",
            },
            primary_key=["id"],
            indexes=[
                IndexDefinition("idx_audit_table", "audit_log", ["table_name"]),
                IndexDefinition(
                    "idx_audit_record", "audit_log", ["table_name", "record_id"]
                ),
                IndexDefinition("idx_audit_time", "audit_log", ["changed_at"]),
                IndexDefinition("idx_audit_user", "audit_log", ["changed_by"]),
            ],
            estimated_rows=10000000,
        )

    def _create_performance_views(self):
        """Create materialized views for common performance queries."""

        # Robot performance summary
        self.views[
            "robot_performance_summary"
        ] = """
        CREATE VIEW robot_performance_summary AS
        SELECT 
            r.id as robot_id,
            r.name as robot_name,
            COUNT(DISTINCT ls.id) as total_sessions,
            COUNT(DISTINCT le.id) as total_episodes,
            COUNT(DISTINCT CASE WHEN le.outcome = 'success' THEN le.id END) as successful_episodes,
            AVG(le.total_reward) as avg_episode_reward,
            MAX(le.total_reward) as max_episode_reward,
            COUNT(DISTINCT rs.skill_type) as skills_learned,
            AVG(rs.proficiency_score) as avg_skill_proficiency,
            MAX(ls.updated_at) as last_training_session
        FROM robots r
        LEFT JOIN learning_sessions ls ON r.id = ls.robot_id
        LEFT JOIN learning_episodes le ON ls.id = le.session_id
        LEFT JOIN robot_skills rs ON r.id = rs.robot_id
        GROUP BY r.id, r.name
        """

        # Training progress by curriculum stage
        self.views[
            "curriculum_progress_view"
        ] = """
        CREATE VIEW curriculum_progress_view AS
        SELECT 
            cs.plan_id,
            cs.stage_order,
            cs.name as stage_name,
            COUNT(DISTINCT ls.id) as sessions_using_stage,
            COUNT(DISTINCT le.id) as episodes_in_stage,
            AVG(le.total_reward) as avg_stage_reward,
            COUNT(DISTINCT CASE WHEN le.outcome = 'success' THEN le.id END) * 100.0 / 
                COUNT(DISTINCT le.id) as success_rate_percent
        FROM curriculum_stages cs
        LEFT JOIN learning_sessions ls ON cs.plan_id = ls.curriculum_plan_id
        LEFT JOIN learning_episodes le ON ls.id = le.session_id 
            AND le.target_skill_type IN (
                SELECT skill_type FROM stage_skills WHERE stage_id = cs.id
            )
        GROUP BY cs.plan_id, cs.stage_order, cs.name
        ORDER BY cs.plan_id, cs.stage_order
        """

        # Recent performance trends
        self.views[
            "recent_performance_trends"
        ] = """
        CREATE VIEW recent_performance_trends AS
        SELECT 
            DATE(le.start_time) as training_date,
            COUNT(DISTINCT le.id) as episodes_count,
            AVG(le.total_reward) as avg_reward,
            AVG(le.step_count) as avg_steps,
            COUNT(DISTINCT CASE WHEN le.outcome = 'success' THEN le.id END) * 100.0 / 
                COUNT(DISTINCT le.id) as success_rate_percent,
            COUNT(DISTINCT le.session_id) as active_sessions
        FROM learning_episodes le
        WHERE le.start_time >= datetime('now', '-30 days')
        GROUP BY DATE(le.start_time)
        ORDER BY training_date DESC
        """

    def _create_audit_triggers(self):
        """Create triggers for automatic audit logging."""

        # Audit trigger for robots table
        self.triggers[
            "audit_robots_changes"
        ] = """
        CREATE TRIGGER IF NOT EXISTS audit_robots_changes
        AFTER UPDATE ON robots
        FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (
                table_name, record_id, operation, old_values, new_values, changed_at
            ) VALUES (
                'robots', 
                NEW.id, 
                'UPDATE',
                json_object('name', OLD.name, 'status', OLD.status, 'updated_at', OLD.updated_at),
                json_object('name', NEW.name, 'status', NEW.status, 'updated_at', NEW.updated_at),
                CURRENT_TIMESTAMP
            );
        END
        """

        # Audit trigger for learning sessions
        self.triggers[
            "audit_sessions_changes"
        ] = """
        CREATE TRIGGER IF NOT EXISTS audit_sessions_changes
        AFTER UPDATE ON learning_sessions
        FOR EACH ROW
        BEGIN
            INSERT INTO audit_log (
                table_name, record_id, operation, old_values, new_values, changed_at
            ) VALUES (
                'learning_sessions',
                NEW.id,
                'UPDATE', 
                json_object('status', OLD.status, 'total_episodes', OLD.total_episodes, 
                           'total_reward', OLD.total_reward),
                json_object('status', NEW.status, 'total_episodes', NEW.total_episodes,
                           'total_reward', NEW.total_reward),
                CURRENT_TIMESTAMP
            );
        END
        """

    def get_migration_sql(self) -> List[str]:
        """Generate SQL statements for database migration."""
        statements = []

        # Create tables
        for table in self.tables.values():
            statements.append(self._generate_table_sql(table))

        # Create indexes
        for table in self.tables.values():
            for index in table.indexes:
                statements.append(index.to_sql())

        # Create views
        for view_sql in self.views.values():
            statements.append(view_sql)

        # Create triggers
        for trigger_sql in self.triggers.values():
            statements.append(trigger_sql)

        return statements

    def _generate_table_sql(self, table: TableDefinition) -> str:
        """Generate CREATE TABLE SQL for a table definition."""
        columns = []

        # Add column definitions
        for col_name, col_type in table.columns.items():
            columns.append(f"    {col_name} {col_type}")

        # Add foreign key constraints
        for col_name, ref_table_col in table.foreign_keys.items():
            ref_table, ref_col = ref_table_col.split(".")
            columns.append(
                f"    FOREIGN KEY ({col_name}) REFERENCES {ref_table}({ref_col})"
            )

        # Add check constraints
        for constraint in table.constraints:
            columns.append(f"    {constraint}")

        columns_sql = ",\n".join(columns)

        return f"""
CREATE TABLE IF NOT EXISTS {table.name} (
{columns_sql}
)"""

    def get_performance_analysis(self) -> Dict[str, any]:
        """Analyze schema performance characteristics."""
        return {
            "total_tables": len(self.tables),
            "total_indexes": sum(len(table.indexes) for table in self.tables.values()),
            "estimated_total_rows": sum(
                table.estimated_rows or 0 for table in self.tables.values()
            ),
            "large_tables": [
                table.name
                for table in self.tables.values()
                if (table.estimated_rows or 0) > 1000000
            ],
            "optimization_recommendations": self._get_optimization_recommendations(),
        }

    def _get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations."""
        recommendations = []

        # Check for tables that might need partitioning
        large_tables = [
            t for t in self.tables.values() if (t.estimated_rows or 0) > 1000000
        ]
        if large_tables:
            recommendations.append(
                f"Consider partitioning large tables: {[t.name for t in large_tables]}"
            )

        # Check for missing indexes on foreign keys
        for table in self.tables.values():
            for fk_col in table.foreign_keys.keys():
                has_index = any(fk_col in idx.columns for idx in table.indexes)
                if not has_index:
                    recommendations.append(
                        f"Add index on foreign key {table.name}.{fk_col}"
                    )

        # Suggest connection pooling for high-volume tables
        if any((t.estimated_rows or 0) > 10000000 for t in self.tables.values()):
            recommendations.append(
                "Implement connection pooling for high-volume operations"
            )

        return recommendations
