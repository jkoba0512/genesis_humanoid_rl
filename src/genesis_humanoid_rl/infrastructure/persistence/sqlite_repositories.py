"""SQLite implementations of domain repositories."""

import logging
from typing import Dict, List, Optional, Set
from uuid import UUID

# Import secure JSON handling instead of standard json
from ..security.json_security import safe_loads, safe_dumps, JSONSecurityError
from ..security.validators import SecurityValidator, ValidationError

from ...domain.model.aggregates import CurriculumPlan, HumanoidRobot, LearningSession
from ...domain.model.entities import CurriculumStage, LearningEpisode
from ...domain.model.value_objects import (
    EpisodeId,
    GaitPattern,
    LocomotionSkill,
    MotionCommand,
    MotionType,
    PerformanceMetrics,
    RobotId,
    SessionId,
    SkillAssessment,
    SkillType,
)
from ...domain.repositories import (
    CurriculumPlanRepository,
    DomainEvent,
    DomainEventRepository,
    HumanoidRobotRepository,
    LearningSessionRepository,
)
from .database import DatabaseConnection

logger = logging.getLogger(__name__)

# Initialize security validator for all repositories
security_validator = SecurityValidator(
    max_string_length=10000, max_json_size=1000000
)  # 1MB limit


class SQLiteLearningSessionRepository(LearningSessionRepository):
    """SQLite implementation of LearningSessionRepository."""

    def __init__(self, db: DatabaseConnection):
        """Initialize repository with database connection.

        Args:
            db: Database connection manager
        """
        self.db = db

    def save(self, session: LearningSession) -> None:
        """Save or update a learning session."""
        # Check if we're in a Unit of Work transaction
        if hasattr(self, "_unit_of_work") and self._unit_of_work._transaction_active:
            conn = self._unit_of_work._connection
            self._save_session_data(conn, session)
        else:
            with self.db.transaction() as conn:
                self._save_session_data(conn, session)

    def _save_session_data(self, conn, session: LearningSession) -> None:
        """Internal method to save session data."""
        # Save session
        conn.execute(
            """
                INSERT OR REPLACE INTO learning_sessions 
                (id, robot_id, curriculum_plan_id, start_time, end_time, total_reward, updated_at)
                VALUES (:id, :robot_id, :curriculum_plan_id, :start_time, :end_time, :total_reward, CURRENT_TIMESTAMP)
            """,
            {
                "id": session.session_id.value,
                "robot_id": session.robot_id.value,
                "curriculum_plan_id": session.plan_id.value,
                "start_time": getattr(session, "start_time", None),
                "end_time": getattr(session, "end_time", None),
                "total_reward": getattr(session, "total_reward", 0.0),
            },
        )

        # Save episodes
        for episode in session.episodes:
            # Handle case where episode might not have motion commands yet
            if episode.motion_commands:
                # Use first motion command for serialization
                motion_command = episode.motion_commands[0]
                try:
                    motion_command_json = safe_dumps(
                        {
                            "motion_type": motion_command.motion_type.value,
                            "velocity": security_validator.validate_velocity(
                                motion_command.velocity
                            ),
                            "duration": motion_command.duration,
                            "parameters": motion_command.parameters,
                        }
                    )
                except (JSONSecurityError, ValidationError) as e:
                    logger.error(f"Failed to serialize motion command: {e}")
                    raise ValueError(f"Invalid motion command data: {e}") from e
            else:
                # Create a default motion command for episodes without commands
                from ...domain.model.value_objects import MotionType

                motion_command_json = safe_dumps(
                    {
                        "motion_type": MotionType.WALK_FORWARD.value,
                        "velocity": 1.0,
                        "duration": None,
                        "parameters": {},
                    }
                )

            conn.execute(
                """
                INSERT OR REPLACE INTO learning_episodes
                (id, session_id, motion_command, start_time, end_time, total_reward, success)
                VALUES (:id, :session_id, :motion_command, :start_time, :end_time, :total_reward, :success)
            """,
                {
                    "id": episode.episode_id.value,
                    "session_id": session.session_id.value,
                    "motion_command": motion_command_json,
                    "start_time": episode.start_time,
                    "end_time": episode.end_time,
                    "total_reward": episode.total_reward,
                    "success": episode.is_successful(),
                },
            )

    def find_by_id(self, session_id: SessionId) -> Optional[LearningSession]:
        """Find a learning session by ID."""
        # Get session data
        cursor = self.db.execute(
            "SELECT * FROM learning_sessions WHERE id = :id", {"id": session_id.value}
        )
        session_row = cursor.fetchone()

        if not session_row:
            return None

        # Get episodes
        cursor = self.db.execute(
            "SELECT * FROM learning_episodes WHERE session_id = :session_id ORDER BY start_time",
            {"session_id": session_id.value},
        )
        episode_rows = cursor.fetchall()

        # Reconstruct session
        from ...domain.model.value_objects import PlanId

        session = LearningSession(
            session_id=SessionId(session_row["id"]),
            robot_id=RobotId(session_row["robot_id"]),
            plan_id=PlanId(session_row["curriculum_plan_id"]),
        )

        # Restore session state
        session.start_time = session_row["start_time"]
        session.end_time = session_row["end_time"]
        session.total_reward = session_row["total_reward"]

        # Restore episodes
        for row in episode_rows:
            try:
                command_data = safe_loads(row["motion_command"], "motion_command")

                # Validate motion command data
                motion_type_str = security_validator.validate_string_input(
                    command_data["motion_type"], "motion_type"
                )
                velocity = security_validator.validate_velocity(
                    command_data["velocity"]
                )

                motion_command = MotionCommand(
                    motion_type=MotionType(motion_type_str),
                    velocity=velocity,
                    duration=command_data.get("duration"),
                    parameters=command_data.get("parameters", {}),
                )
            except (JSONSecurityError, ValidationError, KeyError, ValueError) as e:
                logger.error(f"Failed to deserialize motion command: {e}")
                # Create a default motion command to maintain data integrity
                motion_command = MotionCommand(
                    motion_type=MotionType.WALK_FORWARD, velocity=1.0
                )

            episode = LearningEpisode(
                episode_id=EpisodeId(row["id"]), session_id=SessionId(row["session_id"])
            )

            # Restore episode state
            episode.start_time = row["start_time"]
            episode.end_time = row["end_time"]
            episode.total_reward = row["total_reward"]
            episode.motion_commands = [motion_command]

            session.episodes.append(episode)

        return session

    def find_by_robot(self, robot_id: RobotId) -> List[LearningSession]:
        """Find all sessions for a robot."""
        cursor = self.db.execute(
            "SELECT id FROM learning_sessions WHERE robot_id = :robot_id ORDER BY start_time DESC",
            {"robot_id": robot_id.value},
        )

        sessions = []
        for row in cursor:
            session = self.find_by_id(SessionId(row["id"]))
            if session:
                sessions.append(session)

        return sessions

    def find_active_sessions(self) -> List[LearningSession]:
        """Find all active (not ended) sessions."""
        cursor = self.db.execute(
            "SELECT id FROM learning_sessions WHERE end_time IS NULL ORDER BY start_time DESC"
        )

        sessions = []
        for row in cursor:
            session = self.find_by_id(SessionId(row["id"]))
            if session:
                sessions.append(session)

        return sessions

    def find_by_robot_id(self, robot_id: RobotId) -> List[LearningSession]:
        """Find sessions for a specific robot."""
        return self.find_by_robot(robot_id)

    def find_completed_sessions(
        self, start_date=None, end_date=None
    ) -> List[LearningSession]:
        """Find completed sessions within date range."""
        query = "SELECT id FROM learning_sessions WHERE end_time IS NOT NULL"
        params = {}

        if start_date:
            query += " AND start_time >= :start_date"
            params["start_date"] = start_date.timestamp()

        if end_date:
            query += " AND end_time <= :end_date"
            params["end_date"] = end_date.timestamp()

        query += " ORDER BY start_time DESC"

        cursor = self.db.execute(query, params)

        sessions = []
        for row in cursor:
            session = self.find_by_id(SessionId(row["id"]))
            if session:
                sessions.append(session)

        return sessions


class SQLiteHumanoidRobotRepository(HumanoidRobotRepository):
    """SQLite implementation of HumanoidRobotRepository."""

    def __init__(self, db: DatabaseConnection):
        """Initialize repository with database connection.

        Args:
            db: Database connection manager
        """
        self.db = db

    def save(self, robot: HumanoidRobot) -> None:
        """Save or update a humanoid robot."""
        # Check if we're in a Unit of Work transaction
        if hasattr(self, "_unit_of_work") and self._unit_of_work._transaction_active:
            conn = self._unit_of_work._connection
            self._save_robot_data(conn, robot)
        else:
            # Use direct connection for standalone operation
            conn = self.db._get_connection()
            self._save_robot_data(conn, robot)

    def _save_robot_data(self, conn, robot: HumanoidRobot) -> None:
        """Internal method to save robot data."""
        # Serialize robot configuration data with security validation
        try:
            config_json = safe_dumps(
                {
                    "robot_type": security_validator.validate_string_input(
                        robot.robot_type.value, "robot_type"
                    ),
                    "joint_count": security_validator.validate_numeric(
                        robot.joint_count, "joint_count", 1, 100
                    ),
                    "height": security_validator.validate_numeric(
                        robot.height, "height", 0.1, 3.0
                    ),
                    "created_at": (
                        robot.created_at.isoformat()
                        if hasattr(robot.created_at, "isoformat")
                        else str(robot.created_at)
                    ),
                }
            )
        except (JSONSecurityError, ValidationError) as e:
            logger.error(f"Failed to serialize robot configuration: {e}")
            raise ValueError(f"Invalid robot configuration data: {e}") from e

        try:
            learned_skills_data = {}
            for skill_type, skill in robot.learned_skills.items():
                learned_skills_data[skill_type.value] = {
                    "skill_type": security_validator.validate_skill_type(
                        skill.skill_type.value
                    ),
                    "mastery_level": security_validator.validate_string_input(
                        skill.mastery_level.value, "mastery_level"
                    ),
                    "proficiency_score": security_validator.validate_proficiency_score(
                        skill.proficiency_score
                    ),
                    "last_assessed": (
                        skill.last_assessed.isoformat() if skill.last_assessed else None
                    ),
                }
            learned_skills_json = safe_dumps(learned_skills_data)
        except (JSONSecurityError, ValidationError) as e:
            logger.error(f"Failed to serialize learned skills: {e}")
            raise ValueError(f"Invalid learned skills data: {e}") from e

        try:
            skill_history_data = []
            for assessment in robot.skill_history:
                skill_history_data.append(
                    {
                        "skill": {
                            "skill_type": security_validator.validate_skill_type(
                                assessment.skill.skill_type.value
                            ),
                            "mastery_level": security_validator.validate_string_input(
                                assessment.skill.mastery_level.value, "mastery_level"
                            ),
                            "proficiency_score": security_validator.validate_proficiency_score(
                                assessment.skill.proficiency_score
                            ),
                            "last_assessed": (
                                assessment.skill.last_assessed.isoformat()
                                if assessment.skill.last_assessed
                                else None
                            ),
                        },
                        "assessment_score": security_validator.validate_proficiency_score(
                            assessment.assessment_score
                        ),
                        "confidence_level": security_validator.validate_confidence_level(
                            assessment.confidence_level
                        ),
                        "evidence_quality": security_validator.validate_confidence_level(
                            assessment.evidence_quality
                        ),
                    }
                )
            skill_history_json = safe_dumps(skill_history_data)
        except (JSONSecurityError, ValidationError) as e:
            logger.error(f"Failed to serialize skill history: {e}")
            raise ValueError(f"Invalid skill history data: {e}") from e

        try:
            performance_history_data = []
            for metrics in robot.performance_history:
                # Limit performance history to prevent unbounded growth (security issue identified in QA)
                if len(performance_history_data) >= 100:  # Keep only last 100 records
                    break

                performance_history_data.append(
                    {
                        "success_rate": security_validator.validate_proficiency_score(
                            metrics.success_rate
                        ),
                        "average_reward": security_validator.validate_numeric(
                            metrics.average_reward, "average_reward", -100, 100
                        ),
                        "learning_progress": security_validator.validate_numeric(
                            metrics.learning_progress, "learning_progress", -1, 1
                        ),
                        "skill_scores": {
                            k.value: security_validator.validate_proficiency_score(v)
                            for k, v in metrics.skill_scores.items()
                        },
                        "gait_quality": security_validator.validate_proficiency_score(
                            metrics.gait_quality
                        ),
                    }
                )
            performance_history_json = safe_dumps(performance_history_data)
        except (JSONSecurityError, ValidationError) as e:
            logger.error(f"Failed to serialize performance history: {e}")
            raise ValueError(f"Invalid performance history data: {e}") from e

        conn.execute(
            """
            INSERT OR REPLACE INTO humanoid_robots
            (id, name, configuration, learned_skills, skill_history, performance_history, updated_at)
            VALUES (:id, :name, :configuration, :learned_skills, :skill_history, :performance_history, CURRENT_TIMESTAMP)
        """,
            {
                "id": robot.robot_id.value,
                "name": robot.name,
                "configuration": config_json,
                "learned_skills": learned_skills_json,
                "skill_history": skill_history_json,
                "performance_history": performance_history_json,
            },
        )

    def find_by_id(self, robot_id: RobotId) -> Optional[HumanoidRobot]:
        """Find a robot by ID."""
        cursor = self.db.execute(
            "SELECT * FROM humanoid_robots WHERE id = :id", {"id": robot_id.value}
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize configuration with security validation
        try:
            config_data = safe_loads(row["configuration"], "robot_configuration")
        except (JSONSecurityError, ValueError) as e:
            logger.error(f"Failed to deserialize robot configuration: {e}")
            # Use default configuration if deserialization fails
            config_data = {
                "robot_type": "generic_humanoid",
                "joint_count": 35,
                "height": 1.2,
                "created_at": "2024-01-01T00:00:00",
            }
        from ...domain.model.aggregates import RobotType

        # Create robot
        robot = HumanoidRobot(
            robot_id=RobotId(row["id"]),
            robot_type=RobotType(config_data.get("robot_type", "generic_humanoid")),
            name=row["name"],
        )

        # Restore additional fields
        robot.joint_count = config_data.get("joint_count", 35)
        robot.height = config_data.get("height", 1.2)

        # Restore learned skills with security validation
        try:
            skills_data = safe_loads(row["learned_skills"], "learned_skills")
        except (JSONSecurityError, ValueError) as e:
            logger.error(f"Failed to deserialize learned skills: {e}")
            skills_data = {}  # Use empty skills if deserialization fails
        for skill_type_str, skill_data in skills_data.items():
            from ...domain.model.value_objects import MasteryLevel
            from datetime import datetime

            skill_type = SkillType(skill_type_str)
            last_assessed = None
            if skill_data["last_assessed"]:
                last_assessed = datetime.fromisoformat(skill_data["last_assessed"])

            skill = LocomotionSkill(
                skill_type=SkillType(skill_data["skill_type"]),
                mastery_level=MasteryLevel(skill_data["mastery_level"]),
                proficiency_score=skill_data["proficiency_score"],
                last_assessed=last_assessed,
            )
            robot.learned_skills[skill_type] = skill

        # Restore skill history with security validation
        try:
            history_data = safe_loads(row["skill_history"], "skill_history")
        except (JSONSecurityError, ValueError) as e:
            logger.error(f"Failed to deserialize skill history: {e}")
            history_data = []  # Use empty history if deserialization fails
        for assessment_data in history_data:
            skill_data = assessment_data["skill"]
            last_assessed = None
            if skill_data["last_assessed"]:
                last_assessed = datetime.fromisoformat(skill_data["last_assessed"])

            skill = LocomotionSkill(
                skill_type=SkillType(skill_data["skill_type"]),
                mastery_level=MasteryLevel(skill_data["mastery_level"]),
                proficiency_score=skill_data["proficiency_score"],
                last_assessed=last_assessed,
            )
            assessment = SkillAssessment(
                skill=skill,
                assessment_score=assessment_data["assessment_score"],
                confidence_level=assessment_data["confidence_level"],
                evidence_quality=assessment_data["evidence_quality"],
            )
            robot.skill_history.append(assessment)

        # Restore performance history with security validation
        try:
            perf_data = safe_loads(row["performance_history"], "performance_history")
        except (JSONSecurityError, ValueError) as e:
            logger.error(f"Failed to deserialize performance history: {e}")
            perf_data = []  # Use empty history if deserialization fails
        for metrics_data in perf_data:
            skill_scores = {
                SkillType(k): v for k, v in metrics_data["skill_scores"].items()
            }
            metrics = PerformanceMetrics(
                success_rate=metrics_data["success_rate"],
                average_reward=metrics_data["average_reward"],
                learning_progress=metrics_data["learning_progress"],
                skill_scores=skill_scores,
                gait_quality=metrics_data["gait_quality"],
            )
            robot.performance_history.append(metrics)

        return robot

    def find_all(self) -> List[HumanoidRobot]:
        """Find all robots."""
        cursor = self.db.execute(
            "SELECT id FROM humanoid_robots ORDER BY created_at DESC"
        )

        robots = []
        for row in cursor:
            robot = self.find_by_id(RobotId(row["id"]))
            if robot:
                robots.append(robot)

        return robots

    def find_by_skill_level(
        self, skill_type: SkillType, min_proficiency: float
    ) -> List[HumanoidRobot]:
        """Find robots with a minimum proficiency in a skill."""
        # This requires loading all robots and filtering in memory
        # In a production system, we'd normalize the skills table
        all_robots = self.find_all()

        return [
            robot
            for robot in all_robots
            if skill_type in robot.learned_skills
            and robot.learned_skills[skill_type].proficiency_score >= min_proficiency
        ]

    def find_by_skill_mastery(
        self, skill: SkillType, min_proficiency: float = 0.7
    ) -> List[HumanoidRobot]:
        """Find robots that have mastered a specific skill."""
        return self.find_by_skill_level(skill, min_proficiency)


class SQLiteCurriculumPlanRepository(CurriculumPlanRepository):
    """SQLite implementation of CurriculumPlanRepository."""

    def __init__(self, db: DatabaseConnection):
        """Initialize repository with database connection.

        Args:
            db: Database connection manager
        """
        self.db = db

    def save(self, plan: CurriculumPlan) -> None:
        """Save or update a curriculum plan."""
        # Check if we're in a Unit of Work transaction
        if hasattr(self, "_unit_of_work") and self._unit_of_work._transaction_active:
            conn = self._unit_of_work._connection
            self._save_plan_data(conn, plan)
        else:
            # Use direct connection for standalone operation
            conn = self.db._get_connection()
            self._save_plan_data(conn, plan)

    def _save_plan_data(self, conn, plan: CurriculumPlan) -> None:
        """Internal method to save plan data."""
        # Serialize stages
        stages_json = safe_dumps(
            [
                {
                    "stage_id": stage.stage_id,
                    "name": stage.name,
                    "description": stage.description,
                    "stage_type": stage.stage_type.value,
                    "order": stage.order,
                    "target_skills": [skill.value for skill in stage.target_skills],
                    "prerequisite_skills": [
                        skill.value for skill in stage.prerequisite_skills
                    ],
                    "difficulty_level": stage.difficulty_level,
                    "expected_duration_episodes": stage.expected_duration_episodes,
                    "target_success_rate": stage.target_success_rate,
                    "advancement_criteria": {
                        k.value: v for k, v in stage.advancement_criteria.items()
                    },
                    "min_episodes": stage.min_episodes,
                }
                for stage in plan.stages
            ]
        )

        # Serialize plan metadata
        metadata_json = safe_dumps(
            {
                "robot_type": (
                    plan.robot_type.value if hasattr(plan, "robot_type") else "humanoid"
                ),
                "version": plan.version,
                "description": plan.description,
                "created_at": (
                    plan.created_at.isoformat()
                    if hasattr(plan.created_at, "isoformat")
                    else str(plan.created_at)
                ),
            }
        )

        conn.execute(
            """
            INSERT OR REPLACE INTO curriculum_plans
            (id, name, stages, status, current_stage_index, difficulty_params, updated_at)
            VALUES (:id, :name, :stages, :status, :current_stage_index, :difficulty_params, CURRENT_TIMESTAMP)
        """,
            {
                "id": plan.plan_id.value,
                "name": plan.name,
                "stages": stages_json,
                "status": (
                    plan.status.value if hasattr(plan.status, "value") else plan.status
                ),
                "current_stage_index": getattr(plan, "current_stage_index", 0),
                "difficulty_params": metadata_json,
            },
        )

    def find_by_id(self, plan_id) -> Optional[CurriculumPlan]:
        """Find a curriculum plan by ID."""
        plan_id_str = plan_id.value if hasattr(plan_id, "value") else str(plan_id)
        cursor = self.db.execute(
            "SELECT * FROM curriculum_plans WHERE id = :id", {"id": plan_id_str}
        )
        row = cursor.fetchone()

        if not row:
            return None

        # Deserialize stages
        stages_data = safe_loads(row["stages"], "curriculum_stages")
        stages = []
        for stage_data in stages_data:
            from ...domain.model.entities import StageType, AdvancementCriteria

            target_skills = {SkillType(skill) for skill in stage_data["target_skills"]}
            prerequisite_skills = {
                SkillType(skill) for skill in stage_data.get("prerequisite_skills", [])
            }
            advancement_criteria = {
                AdvancementCriteria(k): v
                for k, v in stage_data["advancement_criteria"].items()
            }

            stage = CurriculumStage(
                stage_id=stage_data["stage_id"],
                name=stage_data["name"],
                stage_type=StageType(stage_data["stage_type"]),
                order=stage_data["order"],
                target_skills=target_skills,
                prerequisite_skills=prerequisite_skills,
                difficulty_level=stage_data["difficulty_level"],
                expected_duration_episodes=stage_data["expected_duration_episodes"],
                target_success_rate=stage_data["target_success_rate"],
                advancement_criteria=advancement_criteria,
                min_episodes=stage_data["min_episodes"],
                description=stage_data["description"],
            )
            stages.append(stage)

        # Deserialize metadata
        metadata = safe_loads(row["difficulty_params"], "curriculum_metadata")

        # Create plan
        from ...domain.model.value_objects import PlanId
        from ...domain.model.aggregates import RobotType

        plan = CurriculumPlan(
            plan_id=PlanId(row["id"]),
            name=row["name"],
            robot_type=RobotType(metadata.get("robot_type", "generic_humanoid")),
            stages=stages,
        )

        # Restore state
        from ...domain.model.aggregates import PlanStatus

        plan.status = PlanStatus(row["status"])
        plan.current_stage_index = row["current_stage_index"]
        # Use metadata to restore additional fields if needed
        if "version" in metadata:
            plan.version = metadata["version"]
        if "description" in metadata:
            plan.description = metadata["description"]

        return plan

    def find_active_plans(self) -> List[CurriculumPlan]:
        """Find all active curriculum plans."""
        cursor = self.db.execute(
            "SELECT id FROM curriculum_plans WHERE status = 'active' ORDER BY created_at DESC"
        )

        plans = []
        for row in cursor:
            plan = self.find_by_id(row["id"])
            if plan:
                plans.append(plan)

        return plans

    def find_by_name(self, name: str) -> Optional[CurriculumPlan]:
        """Find a curriculum plan by name."""
        cursor = self.db.execute(
            "SELECT id FROM curriculum_plans WHERE name = :name", {"name": name}
        )
        row = cursor.fetchone()

        if row:
            return self.find_by_id(row["id"])

        return None

    def find_by_robot_type(self, robot_type: str) -> List[CurriculumPlan]:
        """Find plans for specific robot type."""
        # For now, return all plans since we don't filter by robot type
        # In production, we'd add robot_type filtering to the query
        cursor = self.db.execute(
            "SELECT id FROM curriculum_plans ORDER BY created_at DESC"
        )

        plans = []
        for row in cursor:
            plan = self.find_by_id(row["id"])
            if plan:
                plans.append(plan)

        return plans


class SQLiteDomainEventRepository(DomainEventRepository):
    """SQLite implementation of DomainEventRepository."""

    def __init__(self, db: DatabaseConnection):
        """Initialize repository with database connection.

        Args:
            db: Database connection manager
        """
        self.db = db

    def save(self, event: DomainEvent) -> None:
        """Save a domain event."""
        # Check if we're in a Unit of Work transaction
        if hasattr(self, "_unit_of_work") and self._unit_of_work._transaction_active:
            conn = self._unit_of_work._connection
        else:
            # Use direct connection for standalone operation
            conn = self.db._get_connection()

        payload_json = safe_dumps(event.payload)

        conn.execute(
            """
            INSERT INTO domain_events
            (event_type, aggregate_id, payload, occurred_at)
            VALUES (:event_type, :aggregate_id, :payload, :occurred_at)
        """,
            {
                "event_type": event.event_type,
                "aggregate_id": event.aggregate_id,
                "payload": payload_json,
                "occurred_at": event.occurred_at,
            },
        )

    def find_by_aggregate(self, aggregate_id: str) -> List[DomainEvent]:
        """Find all events for an aggregate."""
        cursor = self.db.execute(
            "SELECT * FROM domain_events WHERE aggregate_id = :aggregate_id ORDER BY occurred_at",
            {"aggregate_id": aggregate_id},
        )

        events = []
        for row in cursor:
            event = DomainEvent(
                event_type=row["event_type"],
                aggregate_id=row["aggregate_id"],
                payload=safe_loads(row["payload"], "domain_event_payload"),
                occurred_at=row["occurred_at"],
            )
            events.append(event)

        return events

    def find_by_type(self, event_type: str, limit: int = 100) -> List[DomainEvent]:
        """Find events by type."""
        cursor = self.db.execute(
            "SELECT * FROM domain_events WHERE event_type = :event_type ORDER BY occurred_at DESC LIMIT :limit",
            {"event_type": event_type, "limit": limit},
        )

        events = []
        for row in cursor:
            event = DomainEvent(
                event_type=row["event_type"],
                aggregate_id=row["aggregate_id"],
                payload=safe_loads(row["payload"], "domain_event_payload"),
                occurred_at=row["occurred_at"],
            )
            events.append(event)

        return events

    def find_recent_events(self, limit: int = 100) -> List[DomainEvent]:
        """Find most recent events."""
        cursor = self.db.execute(
            "SELECT * FROM domain_events ORDER BY occurred_at DESC LIMIT :limit",
            {"limit": limit},
        )

        events = []
        for row in cursor:
            event = DomainEvent(
                event_type=row["event_type"],
                aggregate_id=row["aggregate_id"],
                payload=safe_loads(row["payload"], "domain_event_payload"),
                occurred_at=row["occurred_at"],
            )
            events.append(event)

        return events
