"""
Comprehensive tests for domain entities.
Tests entity lifecycle, business rules, and state transitions.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from src.genesis_humanoid_rl.domain.model.entities import (
    LearningEpisode, EpisodeStatus, EpisodeOutcome,
    CurriculumStage, StageType, AdvancementCriteria
)
from src.genesis_humanoid_rl.domain.model.value_objects import (
    EpisodeId, SessionId, SkillType, MasteryLevel, MotionCommand, MotionType,
    PerformanceMetrics, MovementTrajectory, LocomotionSkill, SkillAssessment
)


class TestLearningEpisode:
    """Test LearningEpisode entity business logic and lifecycle."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.episode_id = EpisodeId.generate()
        self.session_id = SessionId.generate()
        
        self.episode = LearningEpisode(
            episode_id=self.episode_id,
            session_id=self.session_id,
            target_skill=SkillType.FORWARD_WALKING,
            max_steps=100
        )
    
    def test_episode_creation_initial_state(self):
        """Test episode is created in correct initial state."""
        assert self.episode.episode_id == self.episode_id
        assert self.episode.session_id == self.session_id
        assert self.episode.status == EpisodeStatus.PENDING
        assert self.episode.outcome is None
        assert self.episode.total_reward == 0.0
        assert self.episode.step_count == 0
        assert self.episode.target_skill == SkillType.FORWARD_WALKING
        assert len(self.episode.motion_commands) == 0
    
    def test_start_episode_valid(self):
        """Test valid episode start."""
        target_skill = SkillType.STATIC_BALANCE
        
        self.episode.start_episode(target_skill)
        
        assert self.episode.status == EpisodeStatus.RUNNING
        assert self.episode.target_skill == target_skill
        assert self.episode.step_count == 0
        assert self.episode.total_reward == 0.0
        assert self.episode.start_time <= datetime.now()
    
    def test_start_episode_invalid_status(self):
        """Test that starting non-pending episode raises error."""
        # Start episode first
        self.episode.start_episode()
        
        # Try to start again
        with pytest.raises(ValueError, match="Cannot start episode in status"):
            self.episode.start_episode()
    
    def test_add_step_reward_valid(self):
        """Test adding step rewards during running episode."""
        self.episode.start_episode()
        
        self.episode.add_step_reward(1.5)
        assert self.episode.total_reward == 1.5
        assert self.episode.step_count == 1
        
        self.episode.add_step_reward(2.0)
        assert self.episode.total_reward == 3.5
        assert self.episode.step_count == 2
    
    def test_add_step_reward_invalid_status(self):
        """Test that adding reward to non-running episode raises error."""
        # Episode not started
        with pytest.raises(ValueError, match="Cannot add reward to episode in status"):
            self.episode.add_step_reward(1.0)
    
    def test_execute_motion_command_valid(self):
        """Test executing motion commands during episode."""
        self.episode.start_episode()
        
        command = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        self.episode.execute_motion_command(command)
        
        assert len(self.episode.motion_commands) == 1
        assert self.episode.motion_commands[0] == command
        assert 'commands' in self.episode.episode_data
        assert len(self.episode.episode_data['commands']) == 1
    
    def test_execute_motion_command_invalid_status(self):
        """Test that executing command in non-running episode raises error."""
        command = MotionCommand(MotionType.WALK_FORWARD, velocity=1.0)
        
        with pytest.raises(ValueError, match="Cannot execute command in episode status"):
            self.episode.execute_motion_command(command)
    
    def test_complete_episode_valid(self):
        """Test valid episode completion."""
        self.episode.start_episode()
        self.episode.add_step_reward(5.0)
        
        performance_metrics = PerformanceMetrics(
            success_rate=0.8,
            average_reward=2.5
        )
        
        self.episode.complete_episode(EpisodeOutcome.SUCCESS, performance_metrics)
        
        assert self.episode.status == EpisodeStatus.COMPLETED
        assert self.episode.outcome == EpisodeOutcome.SUCCESS
        assert self.episode.performance_metrics == performance_metrics
        assert self.episode.end_time is not None
        assert self.episode.end_time >= self.episode.start_time
    
    def test_complete_episode_invalid_status(self):
        """Test that completing non-running episode raises error."""
        with pytest.raises(ValueError, match="Cannot complete episode in status"):
            self.episode.complete_episode(EpisodeOutcome.SUCCESS)
    
    def test_terminate_episode(self):
        """Test episode termination."""
        self.episode.start_episode()
        
        reason = "Robot fell down"
        self.episode.terminate_episode(reason)
        
        assert self.episode.status == EpisodeStatus.TERMINATED
        assert self.episode.outcome == EpisodeOutcome.TERMINATED_EARLY
        assert self.episode.end_time is not None
        assert self.episode.episode_data['termination_reason'] == reason
    
    def test_terminate_already_completed_episode(self):
        """Test that terminating already completed episode is safe."""
        self.episode.start_episode()
        self.episode.complete_episode(EpisodeOutcome.SUCCESS)
        
        # Should not change status
        self.episode.terminate_episode("Test reason")
        assert self.episode.status == EpisodeStatus.COMPLETED
        assert self.episode.outcome == EpisodeOutcome.SUCCESS
    
    def test_fail_episode(self):
        """Test episode failure handling."""
        self.episode.start_episode()
        
        error_message = "Physics simulation error"
        self.episode.fail_episode(error_message)
        
        assert self.episode.status == EpisodeStatus.FAILED
        assert self.episode.outcome == EpisodeOutcome.ERROR
        assert self.episode.end_time is not None
        assert self.episode.episode_data['error_message'] == error_message
    
    def test_get_duration(self):
        """Test episode duration calculation."""
        # Episode not started
        assert self.episode.get_duration() is None
        
        # Start episode
        start_time = datetime.now()
        with patch('src.genesis_humanoid_rl.domain.model.entities.datetime') as mock_datetime:
            mock_datetime.now.return_value = start_time
            self.episode.start_episode()
        
        # Complete episode after some time
        end_time = start_time + timedelta(minutes=5)
        with patch('src.genesis_humanoid_rl.domain.model.entities.datetime') as mock_datetime:
            mock_datetime.now.return_value = end_time
            self.episode.complete_episode(EpisodeOutcome.SUCCESS)
        
        duration = self.episode.get_duration()
        assert duration == timedelta(minutes=5)
    
    def test_get_average_reward_per_step(self):
        """Test average reward per step calculation."""
        self.episode.start_episode()
        
        # No steps yet
        assert self.episode.get_average_reward_per_step() == 0.0
        
        # Add some rewards
        self.episode.add_step_reward(2.0)
        self.episode.add_step_reward(4.0)
        self.episode.add_step_reward(6.0)
        
        # Total: 12.0, Steps: 3, Average: 4.0
        assert self.episode.get_average_reward_per_step() == 4.0
    
    def test_is_successful(self):
        """Test episode success evaluation."""
        self.episode.start_episode()
        
        # Test successful outcomes
        self.episode.complete_episode(EpisodeOutcome.SUCCESS)
        assert self.episode.is_successful()
        
        # Reset for next test
        self.episode.outcome = EpisodeOutcome.PARTIAL_SUCCESS
        assert self.episode.is_successful()
        
        # Test unsuccessful outcomes
        self.episode.outcome = EpisodeOutcome.FAILURE
        assert not self.episode.is_successful()
        
        self.episode.outcome = EpisodeOutcome.ERROR
        assert not self.episode.is_successful()
    
    def test_achieved_target_skill(self):
        """Test target skill achievement evaluation."""
        self.episode.start_episode(SkillType.FORWARD_WALKING)
        
        # No performance metrics
        assert not self.episode.achieved_target_skill()
        
        # Performance metrics without target skill
        metrics_without_skill = PerformanceMetrics(
            success_rate=0.8,
            average_reward=10.0,
            skill_scores={SkillType.TURNING: 0.8}  # Different skill
        )
        self.episode.complete_episode(EpisodeOutcome.SUCCESS, metrics_without_skill)
        assert not self.episode.achieved_target_skill()
        
        # Performance metrics with low target skill score
        metrics_low_skill = PerformanceMetrics(
            success_rate=0.8,
            average_reward=10.0,
            skill_scores={SkillType.FORWARD_WALKING: 0.5}  # Below threshold
        )
        self.episode.performance_metrics = metrics_low_skill
        assert not self.episode.achieved_target_skill()
        
        # Performance metrics with high target skill score
        metrics_high_skill = PerformanceMetrics(
            success_rate=0.8,
            average_reward=10.0,
            skill_scores={SkillType.FORWARD_WALKING: 0.8}  # Above threshold
        )
        self.episode.performance_metrics = metrics_high_skill
        assert self.episode.achieved_target_skill()
    
    def test_get_complexity_score(self):
        """Test episode complexity score calculation."""
        self.episode.start_episode()
        
        # No commands yet
        assert self.episode.get_complexity_score() == 0.0
        
        # Add commands of varying complexity
        simple_command = MotionCommand(MotionType.BALANCE, velocity=0.5)
        complex_command = MotionCommand(MotionType.TURN_LEFT, velocity=2.0)
        
        self.episode.execute_motion_command(simple_command)
        self.episode.execute_motion_command(complex_command)
        
        expected_complexity = (
            simple_command.get_complexity_score() + 
            complex_command.get_complexity_score()
        ) / 2
        
        assert abs(self.episode.get_complexity_score() - expected_complexity) < 0.001
    
    def test_update_trajectory(self):
        """Test movement trajectory update."""
        positions = [(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8)]
        timestamps = [0.0, 1.0, 2.0]
        
        trajectory = MovementTrajectory(positions=positions, timestamps=timestamps)
        
        self.episode.update_trajectory(trajectory)
        
        assert self.episode.movement_trajectory == trajectory
        assert 'trajectory_metrics' in self.episode.episode_data
        assert 'total_distance' in self.episode.episode_data['trajectory_metrics']
        assert 'average_velocity' in self.episode.episode_data['trajectory_metrics']
        assert 'smoothness_score' in self.episode.episode_data['trajectory_metrics']


class TestCurriculumStage:
    """Test CurriculumStage entity business logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = CurriculumStage(
            stage_id="balance_stage",
            name="Balance Training",
            stage_type=StageType.FOUNDATION,
            order=0,
            target_skills={SkillType.STATIC_BALANCE, SkillType.POSTURAL_CONTROL},
            prerequisite_skills=set(),
            difficulty_level=1.0,
            expected_duration_episodes=20,
            target_success_rate=0.7,
            advancement_criteria={AdvancementCriteria.EPISODE_COUNT: 10},
            min_episodes=5
        )
    
    def test_stage_creation_initial_state(self):
        """Test stage is created in correct initial state."""
        assert self.stage.stage_id == "balance_stage"
        assert self.stage.name == "Balance Training"
        assert self.stage.stage_type == StageType.FOUNDATION
        assert self.stage.order == 0
        assert SkillType.STATIC_BALANCE in self.stage.target_skills
        assert self.stage.episodes_completed == 0
        assert self.stage.successful_episodes == 0
    
    def test_can_advance_insufficient_episodes(self):
        """Test advancement fails with insufficient episodes."""
        # Less than min_episodes (5)
        self.stage.episodes_completed = 3
        self.stage.successful_episodes = 3
        
        assert not self.stage.can_advance()
    
    def test_can_advance_low_success_rate(self):
        """Test advancement fails with low success rate."""
        # Sufficient episodes but low success rate
        self.stage.episodes_completed = 10
        self.stage.successful_episodes = 5  # 50% success rate, below 70% target
        
        assert not self.stage.can_advance()
    
    def test_can_advance_unmastered_skills(self):
        """Test advancement fails with unmastered target skills."""
        # Good episodes and success rate but no skill mastery
        self.stage.episodes_completed = 10
        self.stage.successful_episodes = 8  # 80% success rate
        
        # No skill assessments = skills not mastered
        assert not self.stage.can_advance()
    
    def test_can_advance_success(self):
        """Test successful advancement when all criteria met."""
        # Set up sufficient episodes and success rate
        self.stage.episodes_completed = 10
        self.stage.successful_episodes = 8  # 80% success rate
        
        # Add skill assessments showing mastery
        for skill_type in self.stage.target_skills:
            skill = LocomotionSkill(
                skill_type=skill_type,
                mastery_level=MasteryLevel.INTERMEDIATE,  # Mastered
                proficiency_score=0.8
            )
            assessment = SkillAssessment(
                skill=skill,
                assessment_score=0.8,
                confidence_level=0.8,
                evidence_quality=0.8
            )
            self.stage.skill_assessments[skill_type] = assessment
        
        assert self.stage.can_advance()
    
    def test_add_episode_result_successful(self):
        """Test adding successful episode result."""
        # Create successful episode
        episode = LearningEpisode(
            episode_id=EpisodeId.generate(),
            session_id=SessionId.generate(),
            target_skill=SkillType.STATIC_BALANCE
        )
        episode.start_episode()
        
        # Set up performance metrics with skill score
        performance_metrics = PerformanceMetrics(
            success_rate=0.8,
            average_reward=10.0,
            skill_scores={SkillType.STATIC_BALANCE: 0.75}
        )
        episode.complete_episode(EpisodeOutcome.SUCCESS, performance_metrics)
        
        initial_completed = self.stage.episodes_completed
        initial_successful = self.stage.successful_episodes
        
        self.stage.add_episode_result(episode)
        
        assert self.stage.episodes_completed == initial_completed + 1
        assert self.stage.successful_episodes == initial_successful + 1
        assert SkillType.STATIC_BALANCE in self.stage.skill_assessments
        
        # Check skill assessment was created
        assessment = self.stage.skill_assessments[SkillType.STATIC_BALANCE]
        assert assessment.assessment_score == 0.75
    
    def test_add_episode_result_failed(self):
        """Test adding failed episode result."""
        episode = LearningEpisode(
            episode_id=EpisodeId.generate(),
            session_id=SessionId.generate()
        )
        episode.start_episode()
        episode.complete_episode(EpisodeOutcome.FAILURE)
        
        initial_completed = self.stage.episodes_completed
        initial_successful = self.stage.successful_episodes
        
        self.stage.add_episode_result(episode)
        
        assert self.stage.episodes_completed == initial_completed + 1
        assert self.stage.successful_episodes == initial_successful  # No change
    
    def test_get_success_rate(self):
        """Test success rate calculation."""
        # No episodes completed
        assert self.stage.get_success_rate() == 0.0
        
        # Add some results
        self.stage.episodes_completed = 10
        self.stage.successful_episodes = 7
        
        assert self.stage.get_success_rate() == 0.7
    
    def test_is_skill_mastered(self):
        """Test skill mastery checking."""
        # No assessment = not mastered
        assert not self.stage.is_skill_mastered(SkillType.STATIC_BALANCE)
        
        # Add mastered skill assessment
        mastered_skill = LocomotionSkill(
            skill_type=SkillType.STATIC_BALANCE,
            mastery_level=MasteryLevel.ADVANCED  # Above INTERMEDIATE threshold
        )
        mastered_assessment = SkillAssessment(
            skill=mastered_skill,
            assessment_score=0.8,
            confidence_level=0.8,
            evidence_quality=0.8
        )
        self.stage.skill_assessments[SkillType.STATIC_BALANCE] = mastered_assessment
        
        assert self.stage.is_skill_mastered(SkillType.STATIC_BALANCE)
        assert self.stage.is_skill_mastered(SkillType.STATIC_BALANCE, MasteryLevel.INTERMEDIATE)
        assert not self.stage.is_skill_mastered(SkillType.STATIC_BALANCE, MasteryLevel.EXPERT)
    
    def test_get_progress_percentage(self):
        """Test progress percentage calculation."""
        # Initial state - 0% progress
        initial_progress = self.stage.get_progress_percentage()
        assert initial_progress == 0.0
        
        # Partial progress
        self.stage.episodes_completed = 10  # 50% of expected 20
        self.stage.successful_episodes = 7   # 70% success rate (100% of target)
        
        # Add some skill mastery
        skill = LocomotionSkill(
            skill_type=SkillType.STATIC_BALANCE,
            mastery_level=MasteryLevel.INTERMEDIATE
        )
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.8,
            confidence_level=0.8,
            evidence_quality=0.8
        )
        self.stage.skill_assessments[SkillType.STATIC_BALANCE] = assessment
        
        progress = self.stage.get_progress_percentage()
        
        # Should be between 0 and 100
        assert 0.0 <= progress <= 100.0
        assert progress > initial_progress
    
    def test_get_remaining_requirements(self):
        """Test remaining requirements calculation."""
        # Set up partial completion
        self.stage.episodes_completed = 3     # Below min_episodes (5)
        self.stage.successful_episodes = 2    # 67% success rate, below 70% target
        
        requirements = self.stage.get_remaining_requirements()
        
        assert 'min_episodes' in requirements
        assert requirements['min_episodes'] == 2  # 5 - 3
        
        assert 'success_rate_gap' in requirements
        assert abs(requirements['success_rate_gap'] - 0.033) < 0.01  # 0.7 - 0.667
        
        assert 'unmastered_skills' in requirements
        assert len(requirements['unmastered_skills']) == len(self.stage.target_skills)
    
    def test_advancement_criteria_evaluation(self):
        """Test various advancement criteria evaluation."""
        # Test episode count criteria
        self.stage.advancement_criteria = {AdvancementCriteria.EPISODE_COUNT: 8}
        self.stage.episodes_completed = 10
        self.stage.successful_episodes = 8
        
        # Add skill mastery
        for skill_type in self.stage.target_skills:
            skill = LocomotionSkill(
                skill_type=skill_type,
                mastery_level=MasteryLevel.INTERMEDIATE
            )
            assessment = SkillAssessment(
                skill=skill,
                assessment_score=0.8,
                confidence_level=0.8,
                evidence_quality=0.8
            )
            self.stage.skill_assessments[skill_type] = assessment
        
        assert self.stage.can_advance()
        
        # Test success rate criteria
        self.stage.advancement_criteria = {AdvancementCriteria.SUCCESS_RATE: 0.9}
        assert not self.stage.can_advance()  # Only 80% success rate
        
        # Test skill mastery criteria
        self.stage.advancement_criteria = {AdvancementCriteria.SKILL_MASTERY: 1.0}
        assert self.stage.can_advance()  # 100% of skills mastered