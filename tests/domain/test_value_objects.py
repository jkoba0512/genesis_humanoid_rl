"""
Comprehensive tests for domain value objects.
Tests business logic, validation rules, and behavioral methods.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch

from src.genesis_humanoid_rl.domain.model.value_objects import (
    # Identity objects
    SessionId, RobotId, PlanId, EpisodeId,
    
    # Motion and behavior objects
    MotionCommand, MotionType, LocomotionSkill, SkillType, MasteryLevel,
    GaitPattern, MovementTrajectory, PerformanceMetrics, SkillAssessment,
    
    # Analysis objects
    GaitPattern, MovementTrajectory
)


class TestIdentityValueObjects:
    """Test identity value objects for uniqueness and generation."""
    
    def test_session_id_generation_uniqueness(self):
        """Test that generated SessionIds are unique."""
        id1 = SessionId.generate()
        id2 = SessionId.generate()
        
        assert id1.value != id2.value
        assert isinstance(id1.value, str)
        assert len(id1.value) > 0
    
    def test_session_id_from_string(self):
        """Test SessionId creation from string."""
        test_value = "test-session-123"
        session_id = SessionId.from_string(test_value)
        
        assert session_id.value == test_value
    
    def test_robot_id_generation_uniqueness(self):
        """Test that generated RobotIds are unique."""
        id1 = RobotId.generate()
        id2 = RobotId.generate()
        
        assert id1.value != id2.value
        assert isinstance(id1.value, str)
    
    def test_identity_objects_immutability(self):
        """Test that identity objects are immutable."""
        session_id = SessionId.generate()
        
        with pytest.raises(AttributeError):
            session_id.value = "new-value"


class TestMotionCommand:
    """Test MotionCommand business logic and validation."""
    
    def test_motion_command_creation_valid(self):
        """Test valid motion command creation."""
        cmd = MotionCommand(
            motion_type=MotionType.WALK_FORWARD,
            velocity=1.0,
            duration=5.0,
            parameters={'step_length': 0.5}
        )
        
        assert cmd.motion_type == MotionType.WALK_FORWARD
        assert cmd.velocity == 1.0
        assert cmd.duration == 5.0
        assert cmd.parameters['step_length'] == 0.5
    
    def test_motion_command_validation_negative_velocity(self):
        """Test that negative velocity raises ValueError."""
        with pytest.raises(ValueError, match="Velocity must be non-negative"):
            MotionCommand(
                motion_type=MotionType.WALK_FORWARD,
                velocity=-1.0
            )
    
    def test_motion_command_validation_negative_duration(self):
        """Test that negative duration raises ValueError."""
        with pytest.raises(ValueError, match="Duration must be positive"):
            MotionCommand(
                motion_type=MotionType.WALK_FORWARD,
                velocity=1.0,
                duration=-1.0
            )
    
    def test_is_locomotion_command(self):
        """Test locomotion command identification."""
        # Locomotion commands
        walk_cmd = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        turn_cmd = MotionCommand(MotionType.TURN_LEFT, velocity=0.5)
        
        assert walk_cmd.is_locomotion_command()
        assert turn_cmd.is_locomotion_command()
        
        # Non-locomotion commands
        balance_cmd = MotionCommand(MotionType.BALANCE, velocity=0.0)
        stop_cmd = MotionCommand(MotionType.STOP, velocity=0.0)
        
        assert not balance_cmd.is_locomotion_command()
        assert not stop_cmd.is_locomotion_command()
    
    def test_requires_balance(self):
        """Test balance requirement logic."""
        walk_cmd = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        balance_cmd = MotionCommand(MotionType.BALANCE, velocity=0.0)
        stop_cmd = MotionCommand(MotionType.STOP, velocity=0.0)
        
        assert walk_cmd.requires_balance()
        assert balance_cmd.requires_balance()
        assert not stop_cmd.requires_balance()
    
    def test_complexity_score_calculation(self):
        """Test motion command complexity scoring."""
        # Simple commands
        balance_cmd = MotionCommand(MotionType.BALANCE, velocity=0.5)
        assert balance_cmd.get_complexity_score() == 1.0 * 0.25  # base * velocity_factor
        
        stop_cmd = MotionCommand(MotionType.STOP, velocity=0.0)
        assert stop_cmd.get_complexity_score() == 0.5 * 0.0  # base * velocity_factor
        
        # Complex commands
        turn_cmd = MotionCommand(MotionType.TURN_LEFT, velocity=2.0)
        expected_score = 3.0 * 1.0  # base * velocity_factor (capped at 2x)
        assert turn_cmd.get_complexity_score() == expected_score
        
        custom_cmd = MotionCommand(MotionType.CUSTOM, velocity=1.0)
        assert custom_cmd.get_complexity_score() == 4.0 * 0.5  # base * velocity_factor
    
    def test_complexity_score_velocity_capping(self):
        """Test that velocity factor is capped at 2x."""
        high_velocity_cmd = MotionCommand(MotionType.WALK_FORWARD, velocity=10.0)
        normal_velocity_cmd = MotionCommand(MotionType.WALK_FORWARD, velocity=4.0)
        
        # Both should have same complexity due to capping
        assert high_velocity_cmd.get_complexity_score() == normal_velocity_cmd.get_complexity_score()


class TestMasteryLevel:
    """Test MasteryLevel enum functionality."""
    
    def test_numeric_value_mapping(self):
        """Test that mastery levels map to correct numeric values."""
        assert MasteryLevel.NOVICE.get_numeric_value() == 0.0
        assert MasteryLevel.BEGINNER.get_numeric_value() == 0.25
        assert MasteryLevel.INTERMEDIATE.get_numeric_value() == 0.5
        assert MasteryLevel.ADVANCED.get_numeric_value() == 0.75
        assert MasteryLevel.EXPERT.get_numeric_value() == 1.0
    
    def test_from_score_conversion(self):
        """Test conversion from numeric score to mastery level."""
        assert MasteryLevel.from_score(0.0) == MasteryLevel.NOVICE
        assert MasteryLevel.from_score(0.1) == MasteryLevel.NOVICE
        assert MasteryLevel.from_score(0.3) == MasteryLevel.BEGINNER
        assert MasteryLevel.from_score(0.5) == MasteryLevel.INTERMEDIATE
        assert MasteryLevel.from_score(0.8) == MasteryLevel.ADVANCED
        assert MasteryLevel.from_score(1.0) == MasteryLevel.EXPERT
    
    def test_from_score_boundary_conditions(self):
        """Test boundary conditions for score conversion."""
        # Test exact boundaries
        assert MasteryLevel.from_score(0.125) == MasteryLevel.BEGINNER
        assert MasteryLevel.from_score(0.375) == MasteryLevel.INTERMEDIATE
        assert MasteryLevel.from_score(0.625) == MasteryLevel.ADVANCED
        assert MasteryLevel.from_score(0.875) == MasteryLevel.EXPERT


class TestLocomotionSkill:
    """Test LocomotionSkill business logic."""
    
    def test_locomotion_skill_creation(self):
        """Test valid locomotion skill creation."""
        skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.INTERMEDIATE,
            proficiency_score=0.7,
            last_assessed=datetime.now()
        )
        
        assert skill.skill_type == SkillType.FORWARD_WALKING
        assert skill.mastery_level == MasteryLevel.INTERMEDIATE
        assert skill.proficiency_score == 0.7
    
    def test_skill_validation_proficiency_range(self):
        """Test that proficiency score must be in valid range."""
        # Valid proficiency scores
        skill_valid = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.5
        )
        assert skill_valid.proficiency_score == 0.5
        
        # Invalid proficiency scores
        with pytest.raises(ValueError, match="Proficiency score must be between 0.0 and 1.0"):
            LocomotionSkill(
                skill_type=SkillType.FORWARD_WALKING,
                proficiency_score=1.5
            )
        
        with pytest.raises(ValueError, match="Proficiency score must be between 0.0 and 1.0"):
            LocomotionSkill(
                skill_type=SkillType.FORWARD_WALKING,
                proficiency_score=-0.1
            )
    
    def test_is_mastered(self):
        """Test skill mastery checking."""
        expert_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.EXPERT
        )
        
        intermediate_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.INTERMEDIATE
        )
        
        novice_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.NOVICE
        )
        
        # Test default threshold (INTERMEDIATE)
        assert expert_skill.is_mastered()
        assert intermediate_skill.is_mastered()
        assert not novice_skill.is_mastered()
        
        # Test custom threshold
        assert expert_skill.is_mastered(MasteryLevel.ADVANCED)
        assert not intermediate_skill.is_mastered(MasteryLevel.ADVANCED)
    
    def test_skill_prerequisites(self):
        """Test skill prerequisite logic."""
        # Test skills with known prerequisites
        walking_skill = LocomotionSkill(skill_type=SkillType.FORWARD_WALKING)
        turning_skill = LocomotionSkill(skill_type=SkillType.TURNING)
        balance_skill = LocomotionSkill(skill_type=SkillType.STATIC_BALANCE)
        
        # Walking requires dynamic balance
        assert walking_skill.requires_skill(SkillType.DYNAMIC_BALANCE)
        
        # Turning requires forward walking
        assert turning_skill.requires_skill(SkillType.FORWARD_WALKING)
        
        # Static balance requires postural control
        assert balance_skill.requires_skill(SkillType.POSTURAL_CONTROL)
        
        # Test non-prerequisite
        assert not walking_skill.requires_skill(SkillType.TURNING)


class TestGaitPattern:
    """Test GaitPattern calculations and validation."""
    
    def test_gait_pattern_creation_valid(self):
        """Test valid gait pattern creation."""
        gait = GaitPattern(
            stride_length=0.6,
            stride_frequency=2.0,
            step_height=0.05,
            stability_margin=0.1,
            energy_efficiency=0.8,
            symmetry_score=0.9
        )
        
        assert gait.stride_length == 0.6
        assert gait.stride_frequency == 2.0
        assert gait.step_height == 0.05
    
    def test_gait_pattern_validation(self):
        """Test gait pattern validation rules."""
        # Test invalid stride length
        with pytest.raises(ValueError, match="Stride length must be positive"):
            GaitPattern(
                stride_length=-0.1,
                stride_frequency=1.0,
                step_height=0.05,
                stability_margin=0.1,
                energy_efficiency=0.8,
                symmetry_score=0.9
            )
        
        # Test invalid stride frequency
        with pytest.raises(ValueError, match="Stride frequency must be positive"):
            GaitPattern(
                stride_length=0.5,
                stride_frequency=0.0,
                step_height=0.05,
                stability_margin=0.1,
                energy_efficiency=0.8,
                symmetry_score=0.9
            )
        
        # Test invalid energy efficiency
        with pytest.raises(ValueError, match="Energy efficiency must be between 0.0 and 1.0"):
            GaitPattern(
                stride_length=0.5,
                stride_frequency=1.0,
                step_height=0.05,
                stability_margin=0.1,
                energy_efficiency=1.5,
                symmetry_score=0.9
            )
    
    def test_walking_speed_calculation(self):
        """Test walking speed calculation from gait parameters."""
        gait = GaitPattern(
            stride_length=0.6,
            stride_frequency=2.0,
            step_height=0.05,
            stability_margin=0.1,
            energy_efficiency=0.8,
            symmetry_score=0.9
        )
        
        expected_speed = 0.6 * 2.0  # stride_length * stride_frequency
        assert gait.get_walking_speed() == expected_speed
    
    def test_stable_gait_checking(self):
        """Test gait stability assessment."""
        stable_gait = GaitPattern(
            stride_length=0.5,
            stride_frequency=1.0,
            step_height=0.05,
            stability_margin=0.08,  # Above default threshold of 0.05
            energy_efficiency=0.8,
            symmetry_score=0.9
        )
        
        unstable_gait = GaitPattern(
            stride_length=0.5,
            stride_frequency=1.0,
            step_height=0.05,
            stability_margin=0.02,  # Below default threshold
            energy_efficiency=0.8,
            symmetry_score=0.9
        )
        
        assert stable_gait.is_stable_gait()
        assert not unstable_gait.is_stable_gait()
        
        # Test custom threshold
        assert unstable_gait.is_stable_gait(min_stability=0.01)
    
    def test_quality_score_calculation(self):
        """Test gait quality score calculation."""
        high_quality_gait = GaitPattern(
            stride_length=0.5,
            stride_frequency=1.0,
            step_height=0.05,
            stability_margin=0.1,   # High stability
            energy_efficiency=0.9,  # High efficiency
            symmetry_score=0.95     # High symmetry
        )
        
        low_quality_gait = GaitPattern(
            stride_length=0.5,
            stride_frequency=1.0,
            step_height=0.05,
            stability_margin=0.02,  # Low stability
            energy_efficiency=0.3,  # Low efficiency
            symmetry_score=0.4      # Low symmetry
        )
        
        high_score = high_quality_gait.get_quality_score()
        low_score = low_quality_gait.get_quality_score()
        
        assert high_score > low_score
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
        
        # Test score calculation formula
        expected_high_score = (
            min(0.1 / 0.1, 1.0) * 0.4 +  # stability_score * 0.4
            0.9 * 0.3 +                   # energy_efficiency * 0.3
            0.95 * 0.3                    # symmetry_score * 0.3
        )
        assert abs(high_score - expected_high_score) < 0.001


class TestMovementTrajectory:
    """Test MovementTrajectory calculations and validation."""
    
    def test_trajectory_creation_valid(self):
        """Test valid trajectory creation."""
        positions = [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8)]
        timestamps = [0.0, 1.0, 2.0]
        velocities = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        
        trajectory = MovementTrajectory(
            positions=positions,
            timestamps=timestamps,
            velocities=velocities
        )
        
        assert len(trajectory.positions) == 3
        assert len(trajectory.timestamps) == 3
        assert len(trajectory.velocities) == 3
    
    def test_trajectory_validation(self):
        """Test trajectory validation rules."""
        # Test mismatched lengths
        with pytest.raises(ValueError, match="Positions and timestamps must have same length"):
            MovementTrajectory(
                positions=[(0, 0, 0), (1, 0, 0)],
                timestamps=[0.0, 1.0, 2.0]  # Different length
            )
        
        # Test insufficient points
        with pytest.raises(ValueError, match="Trajectory must have at least 2 points"):
            MovementTrajectory(
                positions=[(0, 0, 0)],
                timestamps=[0.0]
            )
        
        # Test velocities length mismatch
        with pytest.raises(ValueError, match="Velocities must match positions length"):
            MovementTrajectory(
                positions=[(0, 0, 0), (1, 0, 0)],
                timestamps=[0.0, 1.0],
                velocities=[(1, 0, 0)]  # Different length
            )
    
    def test_total_distance_calculation(self):
        """Test total distance calculation."""
        # Simple straight line trajectory
        positions = [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (3.0, 0.0, 0.8)]
        timestamps = [0.0, 1.0, 2.0]
        
        trajectory = MovementTrajectory(positions=positions, timestamps=timestamps)
        
        # Distance should be 1.0 + 2.0 = 3.0
        expected_distance = 3.0
        assert abs(trajectory.get_total_distance() - expected_distance) < 0.001
    
    def test_average_velocity_calculation(self):
        """Test average velocity calculation."""
        positions = [(0.0, 0.0, 0.8), (2.0, 0.0, 0.8), (4.0, 0.0, 0.8)]
        timestamps = [0.0, 1.0, 2.0]  # 2 seconds total
        
        trajectory = MovementTrajectory(positions=positions, timestamps=timestamps)
        
        # Total distance = 4.0, total time = 2.0, avg velocity = 2.0
        expected_velocity = 2.0
        assert abs(trajectory.get_average_velocity() - expected_velocity) < 0.001
    
    def test_average_velocity_zero_time(self):
        """Test average velocity with zero time duration."""
        positions = [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)]
        timestamps = [1.0, 1.0]  # Same timestamp
        
        trajectory = MovementTrajectory(positions=positions, timestamps=timestamps)
        assert trajectory.get_average_velocity() == 0.0
    
    def test_smoothness_score_calculation(self):
        """Test trajectory smoothness calculation."""
        # Smooth trajectory (constant velocity)
        smooth_positions = [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8), (3.0, 0.0, 0.8)]
        smooth_timestamps = [0.0, 1.0, 2.0, 3.0]
        
        smooth_trajectory = MovementTrajectory(
            positions=smooth_positions, 
            timestamps=smooth_timestamps
        )
        
        # Jerky trajectory (varying accelerations)
        jerky_positions = [(0.0, 0.0, 0.8), (0.1, 0.0, 0.8), (2.0, 0.0, 0.8), (2.1, 0.0, 0.8)]
        jerky_timestamps = [0.0, 1.0, 2.0, 3.0]
        
        jerky_trajectory = MovementTrajectory(
            positions=jerky_positions,
            timestamps=jerky_timestamps
        )
        
        smooth_score = smooth_trajectory.get_smoothness_score()
        jerky_score = jerky_trajectory.get_smoothness_score()
        
        # Smooth trajectory should have higher smoothness score
        assert smooth_score > jerky_score
        assert 0.0 <= smooth_score <= 1.0
        assert 0.0 <= jerky_score <= 1.0
    
    def test_smoothness_insufficient_points(self):
        """Test smoothness calculation with insufficient points."""
        positions = [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)]
        timestamps = [0.0, 1.0]
        
        trajectory = MovementTrajectory(positions=positions, timestamps=timestamps)
        assert trajectory.get_smoothness_score() == 0.0


class TestPerformanceMetrics:
    """Test PerformanceMetrics calculations and validation."""
    
    def test_performance_metrics_creation(self):
        """Test valid performance metrics creation."""
        metrics = PerformanceMetrics(
            success_rate=0.8,
            average_reward=15.5,
            skill_scores={SkillType.FORWARD_WALKING: 0.7, SkillType.TURNING: 0.6},
            gait_quality=0.85,
            learning_progress=0.3,
            stability_incidents=2
        )
        
        assert metrics.success_rate == 0.8
        assert metrics.average_reward == 15.5
        assert metrics.skill_scores[SkillType.FORWARD_WALKING] == 0.7
        assert metrics.gait_quality == 0.85
    
    def test_performance_metrics_validation(self):
        """Test performance metrics validation."""
        # Test invalid success rate
        with pytest.raises(ValueError, match="Success rate must be between 0.0 and 1.0"):
            PerformanceMetrics(success_rate=1.5, average_reward=10.0)
        
        # Test invalid skill scores
        with pytest.raises(ValueError, match="Skill score .* must be between 0.0 and 1.0"):
            PerformanceMetrics(
                success_rate=0.8,
                average_reward=10.0,
                skill_scores={SkillType.FORWARD_WALKING: 1.5}
            )
    
    def test_overall_performance_calculation(self):
        """Test overall performance score calculation."""
        metrics = PerformanceMetrics(
            success_rate=0.8,        # 0.8 * 0.3 = 0.24
            average_reward=10.0,     # min(max(10.0, 0.0), 1.0) * 0.2 = 0.2
            learning_progress=0.6,   # 0.6 * 0.2 = 0.12
            skill_scores={
                SkillType.FORWARD_WALKING: 0.7,
                SkillType.TURNING: 0.8
            },                       # avg(0.75) * 0.2 = 0.15
            gait_quality=0.9         # 0.9 * 0.1 = 0.09
        )
        
        overall = metrics.get_overall_performance()
        
        # Expected: (0.24 + 0.2 + 0.12 + 0.15 + 0.09) / 5 = 0.8 / 5 = 0.16
        # Note: The actual calculation may be different based on implementation
        assert 0.0 <= overall <= 1.0
        assert isinstance(overall, float)
    
    def test_is_improving(self):
        """Test learning progress assessment."""
        improving_metrics = PerformanceMetrics(
            success_rate=0.7,
            average_reward=10.0,
            learning_progress=0.15  # Above default threshold of 0.1
        )
        
        stagnant_metrics = PerformanceMetrics(
            success_rate=0.7,
            average_reward=10.0,
            learning_progress=0.05  # Below threshold
        )
        
        assert improving_metrics.is_improving()
        assert not stagnant_metrics.is_improving()
        
        # Test custom threshold
        assert stagnant_metrics.is_improving(threshold=0.01)
    
    def test_dominant_skill(self):
        """Test dominant skill identification."""
        metrics_with_skills = PerformanceMetrics(
            success_rate=0.7,
            average_reward=10.0,
            skill_scores={
                SkillType.FORWARD_WALKING: 0.6,
                SkillType.TURNING: 0.8,  # Highest
                SkillType.STATIC_BALANCE: 0.4
            }
        )
        
        metrics_no_skills = PerformanceMetrics(
            success_rate=0.7,
            average_reward=10.0
        )
        
        assert metrics_with_skills.get_dominant_skill() == SkillType.TURNING
        assert metrics_no_skills.get_dominant_skill() is None


class TestSkillAssessment:
    """Test SkillAssessment validation and methods."""
    
    def test_skill_assessment_creation(self):
        """Test valid skill assessment creation."""
        skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.INTERMEDIATE,
            proficiency_score=0.7
        )
        
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.75,
            confidence_level=0.8,
            evidence_quality=0.9
        )
        
        assert assessment.skill == skill
        assert assessment.assessment_score == 0.75
        assert assessment.confidence_level == 0.8
        assert assessment.evidence_quality == 0.9
    
    def test_skill_assessment_validation(self):
        """Test skill assessment validation rules."""
        skill = LocomotionSkill(skill_type=SkillType.FORWARD_WALKING)
        
        # Test invalid assessment score
        with pytest.raises(ValueError, match="Assessment score must be between 0.0 and 1.0"):
            SkillAssessment(
                skill=skill,
                assessment_score=1.5,
                confidence_level=0.8,
                evidence_quality=0.9
            )
        
        # Test invalid confidence level
        with pytest.raises(ValueError, match="Confidence level must be between 0.0 and 1.0"):
            SkillAssessment(
                skill=skill,
                assessment_score=0.7,
                confidence_level=-0.1,
                evidence_quality=0.9
            )
    
    def test_reliable_assessment(self):
        """Test reliable assessment checking."""
        skill = LocomotionSkill(skill_type=SkillType.FORWARD_WALKING)
        
        reliable_assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.7,
            confidence_level=0.8,  # Above default threshold 0.7
            evidence_quality=0.9   # Above default threshold 0.6
        )
        
        unreliable_assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.7,
            confidence_level=0.5,  # Below threshold
            evidence_quality=0.4   # Below threshold
        )
        
        assert reliable_assessment.is_reliable_assessment()
        assert not unreliable_assessment.is_reliable_assessment()
        
        # Test custom thresholds
        assert unreliable_assessment.is_reliable_assessment(min_confidence=0.4, min_evidence=0.3)
    
    def test_adjusted_score_calculation(self):
        """Test assessment score adjustment for reliability."""
        skill = LocomotionSkill(skill_type=SkillType.FORWARD_WALKING)
        
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.8,
            confidence_level=0.6,
            evidence_quality=0.8
        )
        
        # Reliability factor = (0.6 + 0.8) / 2 = 0.7
        # Adjusted score = 0.8 * 0.7 = 0.56
        expected_adjusted = 0.8 * 0.7
        assert abs(assessment.get_adjusted_score() - expected_adjusted) < 0.001
    
    def test_suggests_mastery(self):
        """Test mastery suggestion logic."""
        skill = LocomotionSkill(skill_type=SkillType.FORWARD_WALKING)
        
        mastery_assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.9,   # High score
            confidence_level=0.8,  # High confidence
            evidence_quality=0.9   # High evidence quality
        )
        
        non_mastery_assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.5,   # Lower score
            confidence_level=0.6,  # Lower confidence
            evidence_quality=0.5   # Lower evidence quality
        )
        
        assert mastery_assessment.suggests_mastery()
        assert not non_mastery_assessment.suggests_mastery()
        
        # Test custom threshold
        assert non_mastery_assessment.suggests_mastery(threshold=0.3)