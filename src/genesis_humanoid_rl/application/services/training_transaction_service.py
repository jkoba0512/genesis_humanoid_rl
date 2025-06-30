"""
Training Transaction Service using Unit of Work pattern.

Demonstrates proper transaction boundary management for complex business operations
that span multiple aggregates and repositories.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from ...domain.model.aggregates import HumanoidRobot, LearningSession
from ...domain.model.entities import EpisodeOutcome
from ...domain.model.value_objects import (
    PerformanceMetrics,
    RobotId,
    SessionId,
    SkillAssessment,
    SkillType,
)
from ...domain.repositories import DomainEvent
from ...infrastructure.persistence.database import DatabaseConnection
from ...infrastructure.persistence.unit_of_work import unit_of_work, UnitOfWorkError

logger = logging.getLogger(__name__)


class TrainingTransactionService:
    """
    Service for managing complex training operations with proper transaction boundaries.

    Uses Unit of Work pattern to ensure atomicity across multiple repository operations.
    Addresses the transaction boundary mismatch identified in Five Whys analysis.
    """

    def __init__(self, db_connection: DatabaseConnection):
        """Initialize service with database connection.

        Args:
            db_connection: Database connection for transactions
        """
        self.db_connection = db_connection

    def complete_episode_with_skill_assessment(
        self,
        session_id: SessionId,
        episode_outcome: EpisodeOutcome,
        performance_metrics: PerformanceMetrics,
        skill_assessments: Optional[Dict[SkillType, SkillAssessment]] = None,
    ) -> Dict[str, Any]:
        """
        Complete an episode and update robot skills atomically.

        This operation spans multiple aggregates and must be atomic:
        1. Update learning session with episode completion
        2. Update robot with new skill assessments
        3. Record domain events for both changes

        Args:
            session_id: ID of the learning session
            episode_outcome: How the episode ended
            performance_metrics: Episode performance data
            skill_assessments: New skill assessments for the robot (optional)

        Returns:
            Dictionary with operation results and statistics

        Raises:
            UnitOfWorkError: If transaction fails and cannot be completed
        """
        try:
            with unit_of_work(self.db_connection) as uow:
                # Load required aggregates
                session = uow.sessions.find_by_id(session_id)
                if not session:
                    raise ValueError(f"Session {session_id.value} not found")

                robot = uow.robots.find_by_id(session.robot_id)
                if not robot:
                    raise ValueError(f"Robot {session.robot_id.value} not found")

                # Store original state for comparison
                original_episode_count = session.total_episodes
                original_robot_skills = len(robot.learned_skills)

                # Complete episode in session (create one if none exists)
                if session.active_episode is None:
                    if session.status.value == "created":
                        session.start_session()
                    episode = session.create_episode()
                    episode.start_episode()

                session.complete_episode(episode_outcome, performance_metrics)

                # Update robot skills if provided
                skills_updated = 0
                if skill_assessments:
                    for skill_type, assessment in skill_assessments.items():
                        robot.assess_skill(skill_type, assessment)
                        skills_updated += 1

                # Add robot performance to history
                robot.performance_history.append(performance_metrics)

                # Save changes to repositories
                uow.sessions.save(session)
                uow.robots.save(robot)

                # Record domain events
                episode_event = DomainEvent(
                    event_type="episode_completed",
                    aggregate_id=session.session_id.value,
                    payload={
                        "session_id": session.session_id.value,
                        "robot_id": robot.robot_id.value,
                        "outcome": episode_outcome.value,
                        "performance": {
                            "success_rate": performance_metrics.success_rate,
                            "average_reward": performance_metrics.average_reward,
                            "learning_progress": performance_metrics.learning_progress,
                        },
                        "skills_updated": skills_updated,
                    },
                    occurred_at=datetime.now().timestamp(),
                )
                uow.events.save(episode_event)

                if skills_updated > 0:
                    skill_event = DomainEvent(
                        event_type="robot_skills_updated",
                        aggregate_id=robot.robot_id.value,
                        payload={
                            "robot_id": robot.robot_id.value,
                            "session_id": session.session_id.value,
                            "skills_count": skills_updated,
                            "total_skills": len(robot.learned_skills),
                        },
                        occurred_at=datetime.now().timestamp(),
                    )
                    uow.events.save(skill_event)

                # Commit all changes atomically
                uow.commit()

                logger.info(
                    f"Episode completion transaction successful: "
                    f"session={session_id.value}, outcome={episode_outcome.value}, "
                    f"skills_updated={skills_updated}"
                )

                return {
                    "success": True,
                    "session_id": session_id.value,
                    "robot_id": robot.robot_id.value,
                    "episode_outcome": episode_outcome.value,
                    "episodes_completed": session.total_episodes,
                    "new_episodes": session.total_episodes - original_episode_count,
                    "skills_updated": skills_updated,
                    "total_robot_skills": len(robot.learned_skills),
                    "new_robot_skills": len(robot.learned_skills)
                    - original_robot_skills,
                    "performance_metrics": {
                        "success_rate": performance_metrics.success_rate,
                        "average_reward": performance_metrics.average_reward,
                        "learning_progress": performance_metrics.learning_progress,
                    },
                }

        except UnitOfWorkError as e:
            logger.error(f"Transaction failed for episode completion: {e}")
            return {
                "success": False,
                "error": "Transaction failed",
                "details": str(e),
                "session_id": session_id.value,
            }
        except Exception as e:
            logger.error(f"Episode completion failed: {e}")
            return {
                "success": False,
                "error": "Operation failed",
                "details": str(e),
                "session_id": session_id.value,
            }

    def transfer_skills_between_robots(
        self,
        source_robot_id: RobotId,
        target_robot_id: RobotId,
        skill_types: list[SkillType],
    ) -> Dict[str, Any]:
        """
        Transfer specific skills from one robot to another atomically.

        This is a complex cross-aggregate operation that must be atomic:
        1. Load skills from source robot
        2. Create assessments for target robot
        3. Update both robots
        4. Record transfer events

        Args:
            source_robot_id: Robot to copy skills from
            target_robot_id: Robot to copy skills to
            skill_types: List of skills to transfer

        Returns:
            Dictionary with transfer results
        """
        try:
            with unit_of_work(self.db_connection) as uow:
                # Load both robots
                source_robot = uow.robots.find_by_id(source_robot_id)
                if not source_robot:
                    raise ValueError(f"Source robot {source_robot_id.value} not found")

                target_robot = uow.robots.find_by_id(target_robot_id)
                if not target_robot:
                    raise ValueError(f"Target robot {target_robot_id.value} not found")

                # Transfer skills
                skills_transferred = 0
                transferred_skills = []

                for skill_type in skill_types:
                    if skill_type in source_robot.learned_skills:
                        source_skill = source_robot.learned_skills[skill_type]

                        # Create assessment for target robot
                        # Reduce proficiency slightly to represent transfer loss
                        from ...domain.model.value_objects import LocomotionSkill
                        from datetime import datetime

                        transferred_skill = LocomotionSkill(
                            skill_type=source_skill.skill_type,
                            mastery_level=source_skill.mastery_level,
                            proficiency_score=source_skill.proficiency_score
                            * 0.9,  # 10% transfer loss
                            last_assessed=datetime.now(),
                        )

                        transfer_assessment = SkillAssessment(
                            skill=transferred_skill,
                            assessment_score=source_skill.proficiency_score
                            * 0.9,  # 10% transfer loss
                            confidence_level=0.8,  # Medium confidence for transferred skills
                            evidence_quality=0.7,  # Good quality from transfer
                        )

                        target_robot.assess_skill(skill_type, transfer_assessment)
                        skills_transferred += 1
                        transferred_skills.append(skill_type.value)

                # Save both robots
                if skills_transferred > 0:
                    uow.robots.save(source_robot)
                    uow.robots.save(target_robot)

                    # Record transfer event
                    transfer_event = DomainEvent(
                        event_type="skills_transferred",
                        aggregate_id=f"{source_robot_id.value}->{target_robot_id.value}",
                        payload={
                            "source_robot_id": source_robot_id.value,
                            "target_robot_id": target_robot_id.value,
                            "skills_transferred": transferred_skills,
                            "transfer_count": skills_transferred,
                        },
                        occurred_at=datetime.now().timestamp(),
                    )
                    uow.events.save(transfer_event)

                # Commit transaction
                uow.commit()

                logger.info(
                    f"Skill transfer completed: {source_robot_id.value} -> {target_robot_id.value}, "
                    f"skills_transferred={skills_transferred}"
                )

                return {
                    "success": True,
                    "source_robot_id": source_robot_id.value,
                    "target_robot_id": target_robot_id.value,
                    "skills_transferred": skills_transferred,
                    "transferred_skills": transferred_skills,
                }

        except UnitOfWorkError as e:
            logger.error(f"Skill transfer transaction failed: {e}")
            return {"success": False, "error": "Transaction failed", "details": str(e)}
        except Exception as e:
            logger.error(f"Skill transfer failed: {e}")
            return {"success": False, "error": "Operation failed", "details": str(e)}

    def batch_update_session_outcomes(
        self, session_outcomes: Dict[SessionId, EpisodeOutcome]
    ) -> Dict[str, Any]:
        """
        Update multiple sessions with their final outcomes atomically.

        Useful for batch processing of completed training sessions.

        Args:
            session_outcomes: Mapping of session IDs to their final outcomes

        Returns:
            Dictionary with batch update results
        """
        try:
            with unit_of_work(self.db_connection) as uow:
                updated_sessions = []
                failed_sessions = []

                for session_id, outcome in session_outcomes.items():
                    try:
                        session = uow.sessions.find_by_id(session_id)
                        if session:
                            # Instead of complete_episode, just update session status
                            # This is a different kind of operation - batch status update
                            if hasattr(session, "status"):
                                # Update session status based on outcome
                                if outcome == EpisodeOutcome.SUCCESS:
                                    session.status = getattr(
                                        session.status.__class__,
                                        "COMPLETED",
                                        "completed",
                                    )
                                elif outcome == EpisodeOutcome.FAILURE:
                                    session.status = getattr(
                                        session.status.__class__, "FAILED", "failed"
                                    )
                                else:
                                    session.status = getattr(
                                        session.status.__class__,
                                        "COMPLETED",
                                        "completed",
                                    )

                            uow.sessions.save(session)
                            updated_sessions.append(session_id.value)
                        else:
                            failed_sessions.append(
                                (session_id.value, "Session not found")
                            )
                    except Exception as e:
                        failed_sessions.append((session_id.value, str(e)))

                # Record batch update event
                if updated_sessions:
                    batch_event = DomainEvent(
                        event_type="batch_sessions_updated",
                        aggregate_id="batch_operation",
                        payload={
                            "updated_count": len(updated_sessions),
                            "failed_count": len(failed_sessions),
                            "updated_sessions": updated_sessions,
                        },
                        occurred_at=datetime.now().timestamp(),
                    )
                    uow.events.save(batch_event)

                # Commit all changes
                uow.commit()

                logger.info(
                    f"Batch session update completed: "
                    f"updated={len(updated_sessions)}, failed={len(failed_sessions)}"
                )

                return {
                    "success": True,
                    "updated_count": len(updated_sessions),
                    "failed_count": len(failed_sessions),
                    "updated_sessions": updated_sessions,
                    "failed_sessions": failed_sessions,
                }

        except UnitOfWorkError as e:
            logger.error(f"Batch update transaction failed: {e}")
            return {"success": False, "error": "Transaction failed", "details": str(e)}
        except Exception as e:
            logger.error(f"Batch update failed: {e}")
            return {"success": False, "error": "Operation failed", "details": str(e)}
