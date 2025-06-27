"""
Domain model test fixtures and builders.

Provides factory functions and fixtures for creating domain objects
with sensible defaults and configurable parameters.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from unittest.mock import Mock

from src.genesis_humanoid_rl.domain.model.value_objects import (
    # Identity objects
    SessionId, RobotId, PlanId, EpisodeId,
    
    # Motion and behavior objects
    MotionCommand, MotionType, LocomotionSkill, SkillType, MasteryLevel,
    GaitPattern, MovementTrajectory, PerformanceMetrics, SkillAssessment
)

from src.genesis_humanoid_rl.domain.model.entities import (
    LearningEpisode, CurriculumStage, EpisodeOutcome
)

from src.genesis_humanoid_rl.domain.model.aggregates import (
    HumanoidRobot, LearningSession, CurriculumPlan, RobotType
)


class DomainObjectBuilder:
    """
    Builder pattern for creating domain objects with sensible defaults.
    
    Provides a fluent interface for building complex domain objects
    with configurable parameters and automatic validation.
    """
    
    @staticmethod
    def robot(
        robot_id: Optional[str] = None,
        robot_type: RobotType = RobotType.UNITREE_G1,
        name: Optional[str] = None,
        joint_count: int = 35,
        height: float = 1.2,
        weight: float = 35.0
    ) -> HumanoidRobot:
        """Build a HumanoidRobot with sensible defaults."""
        robot_id = robot_id or f"robot-{datetime.now().timestamp()}"
        name = name or f"Test Robot {robot_id}"
        
        return HumanoidRobot(
            robot_id=RobotId.from_string(robot_id),
            robot_type=robot_type,
            name=name,
            joint_count=joint_count,
            height=height,
            weight=weight
        )
    
    @staticmethod
    def learning_session(
        session_id: Optional[str] = None,
        robot_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        created_at: Optional[datetime] = None
    ) -> LearningSession:
        """Build a LearningSession with sensible defaults."""
        session_id = session_id or f"session-{datetime.now().timestamp()}"
        robot_id = robot_id or f"robot-{datetime.now().timestamp()}"
        plan_id = plan_id or f"plan-{datetime.now().timestamp()}"
        created_at = created_at or datetime.now()
        
        return LearningSession(
            session_id=SessionId.from_string(session_id),
            robot_id=RobotId.from_string(robot_id),
            plan_id=PlanId.from_string(plan_id),
            created_at=created_at
        )
    
    @staticmethod
    def learning_episode(
        episode_id: Optional[str] = None,
        session_id: Optional[str] = None,
        step_count: int = 1000,
        total_reward: float = 85.5,
        outcome: EpisodeOutcome = EpisodeOutcome.SUCCESS
    ) -> LearningEpisode:
        """Build a LearningEpisode with sensible defaults."""
        episode_id = episode_id or f"episode-{datetime.now().timestamp()}"
        session_id = session_id or f"session-{datetime.now().timestamp()}"
        
        return LearningEpisode(
            episode_id=EpisodeId.from_string(episode_id),
            session_id=SessionId.from_string(session_id),
            step_count=step_count,
            total_reward=total_reward,
            outcome=outcome,
            start_time=datetime.now()
        )
    
    @staticmethod
    def curriculum_stage(
        stage_id: str = "balance",
        name: str = "Balance Stage",
        stage_type = None,
        order: int = 1,
        min_episodes: int = 20
    ) -> CurriculumStage:
        """Build a CurriculumStage with sensible defaults."""
        from src.genesis_humanoid_rl.domain.model.entities import StageType
        
        if stage_type is None:
            stage_type = StageType.FOUNDATION
            
        return CurriculumStage(
            stage_id=stage_id,
            name=name,
            stage_type=stage_type,
            order=order,
            min_episodes=min_episodes
        )
    
    @staticmethod
    def motion_command(
        motion_type: MotionType = MotionType.WALK_FORWARD,
        velocity: float = 1.0,
        duration: float = 5.0
    ) -> MotionCommand:
        """Build a MotionCommand with sensible defaults."""
        return MotionCommand(
            motion_type=motion_type,
            velocity=velocity,
            duration=duration
        )
    
    @staticmethod
    def locomotion_skill(
        skill_type: SkillType = SkillType.FORWARD_WALKING,
        mastery_level: MasteryLevel = MasteryLevel.INTERMEDIATE,
        proficiency_score: float = 0.7
    ) -> LocomotionSkill:
        """Build a LocomotionSkill with sensible defaults."""
        return LocomotionSkill(
            skill_type=skill_type,
            mastery_level=mastery_level,
            proficiency_score=proficiency_score
        )
    
    @staticmethod
    def performance_metrics(
        success_rate: float = 0.8,
        average_reward: float = 15.5,
        episode_count: int = 100,
        learning_rate: float = 0.02
    ) -> PerformanceMetrics:
        """Build PerformanceMetrics with sensible defaults."""
        return PerformanceMetrics(
            success_rate=success_rate,
            average_reward=average_reward,
            episode_count=episode_count,
            learning_rate=learning_rate
        )
    
    @staticmethod
    def gait_pattern(
        stride_length: float = 0.5,
        stride_frequency: float = 1.2,
        stability_score: float = 0.8,
        symmetry_score: float = 0.9
    ) -> GaitPattern:
        """Build a GaitPattern with sensible defaults."""
        return GaitPattern(
            stride_length=stride_length,
            stride_frequency=stride_frequency,
            stability_score=stability_score,
            symmetry_score=symmetry_score
        )
    
    @staticmethod
    def movement_trajectory(
        positions: Optional[List[np.ndarray]] = None,
        time_points: Optional[List[float]] = None,
        smoothness_score: Optional[float] = None
    ) -> MovementTrajectory:
        """Build a MovementTrajectory with sensible defaults."""
        if positions is None:
            # Create a simple forward walking trajectory
            positions = [
                np.array([0.0, 0.0, 0.8]),
                np.array([0.5, 0.0, 0.8]),
                np.array([1.0, 0.0, 0.8]),
                np.array([1.5, 0.0, 0.8]),
                np.array([2.0, 0.0, 0.8])
            ]
        
        if time_points is None:
            time_points = [0.0, 1.0, 2.0, 3.0, 4.0]
        
        return MovementTrajectory(
            positions=positions,
            time_points=time_points,
            smoothness_score=smoothness_score
        )


# Pytest fixtures using the builder
@pytest.fixture
def domain_builder():
    """Pytest fixture for domain object builder."""
    return DomainObjectBuilder()


@pytest.fixture
def sample_robot(domain_builder):
    """Pytest fixture for a sample robot."""
    return domain_builder.robot()


@pytest.fixture
def sample_learning_session(domain_builder):
    """Pytest fixture for a sample learning session."""
    return domain_builder.learning_session()


@pytest.fixture
def sample_learning_episode(domain_builder):
    """Pytest fixture for a sample learning episode."""
    return domain_builder.learning_episode()


@pytest.fixture
def sample_curriculum_stage(domain_builder):
    """Pytest fixture for a sample curriculum stage."""
    return domain_builder.curriculum_stage()


@pytest.fixture
def sample_motion_command(domain_builder):
    """Pytest fixture for a sample motion command."""
    return domain_builder.motion_command()


@pytest.fixture
def sample_locomotion_skill(domain_builder):
    """Pytest fixture for a sample locomotion skill."""
    return domain_builder.locomotion_skill()


@pytest.fixture
def sample_performance_metrics(domain_builder):
    """Pytest fixture for sample performance metrics."""
    return domain_builder.performance_metrics()


@pytest.fixture
def sample_gait_pattern(domain_builder):
    """Pytest fixture for a sample gait pattern."""
    return domain_builder.gait_pattern()


@pytest.fixture
def sample_movement_trajectory(domain_builder):
    """Pytest fixture for a sample movement trajectory."""
    return domain_builder.movement_trajectory()


# Complex scenario fixtures
@pytest.fixture
def training_scenario():
    """
    Pytest fixture for a complete training scenario.
    
    Creates a robot, session, episodes, and curriculum for integration testing.
    """
    builder = DomainObjectBuilder()
    
    # Create related objects
    robot = builder.robot(robot_id="scenario-robot")
    session = builder.learning_session(
        session_id="scenario-session",
        robot_id="scenario-robot"
    )
    
    episodes = [
        builder.learning_episode(
            episode_id=f"scenario-episode-{i}",
            session_id="scenario-session",
            step_count=1000 + (i * 100),
            total_reward=50.0 + (i * 10.0)
        )
        for i in range(5)
    ]
    
    curriculum_stages = [
        builder.curriculum_stage("balance", "Balance Stage", order=1, min_episodes=20),
        builder.curriculum_stage("small_steps", "Small Steps Stage", order=2, min_episodes=30),
        builder.curriculum_stage("walking", "Walking Stage", order=3, min_episodes=50)
    ]
    
    return {
        'robot': robot,
        'session': session,
        'episodes': episodes,
        'curriculum_stages': curriculum_stages
    }


@pytest.fixture
def skill_progression_scenario():
    """
    Pytest fixture for skill progression testing.
    
    Creates a progression of skills from beginner to expert level.
    """
    builder = DomainObjectBuilder()
    
    skill_progression = []
    for level in [MasteryLevel.BEGINNER, MasteryLevel.INTERMEDIATE, MasteryLevel.EXPERT]:
        for skill_type in [SkillType.STATIC_BALANCE, SkillType.FORWARD_WALKING, SkillType.TURNING]:
            skill = builder.locomotion_skill(
                skill_type=skill_type,
                mastery_level=level,
                proficiency_score=0.3 + (level.get_numeric_value() * 0.3)  # Progressive proficiency
            )
            skill_progression.append(skill)
    
    return skill_progression


# Mock domain service fixtures
@pytest.fixture
def mock_movement_analyzer():
    """Mock movement quality analyzer for testing."""
    mock_analyzer = Mock()
    mock_analyzer.analyze_gait_stability.return_value = 0.8
    mock_analyzer.detect_movement_anomalies.return_value = []
    mock_analyzer.calculate_energy_efficiency.return_value = 0.75
    mock_analyzer.assess_balance_quality.return_value = 0.9
    return mock_analyzer


@pytest.fixture
def mock_curriculum_service():
    """Mock curriculum progression service for testing."""
    mock_service = Mock()
    mock_service.evaluate_advancement_readiness.return_value = True
    mock_service.calculate_difficulty_adjustment.return_value = 1.2
    mock_service.recommend_next_stage.return_value = "walking"
    return mock_service


# Validation utilities
class DomainObjectValidator:
    """Utilities for validating domain objects in tests."""
    
    @staticmethod
    def assert_valid_robot(robot: HumanoidRobot):
        """Assert that robot has valid properties."""
        assert robot.robot_id is not None
        assert robot.robot_type in RobotType
        assert robot.joint_count > 0
        assert robot.height > 0
        assert robot.weight > 0
        assert len(robot.name) > 0
    
    @staticmethod
    def assert_valid_session(session: LearningSession):
        """Assert that learning session has valid properties."""
        assert session.session_id is not None
        assert session.robot_id is not None
        assert session.plan_id is not None
        assert session.start_time is not None
    
    @staticmethod
    def assert_valid_episode(episode: LearningEpisode):
        """Assert that learning episode has valid properties."""
        assert episode.episode_id is not None
        assert episode.session_id is not None
        assert episode.step_count >= 0
        assert episode.outcome in EpisodeOutcome
        assert episode.start_time is not None
    
    @staticmethod
    def assert_valid_performance_metrics(metrics: PerformanceMetrics):
        """Assert that performance metrics are valid."""
        assert 0.0 <= metrics.success_rate <= 1.0
        assert metrics.episode_count >= 0
        assert metrics.learning_rate >= 0


@pytest.fixture
def domain_validator():
    """Pytest fixture for domain object validator."""
    return DomainObjectValidator()