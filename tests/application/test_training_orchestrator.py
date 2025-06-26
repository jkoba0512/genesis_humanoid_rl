"""
Comprehensive tests for application layer training orchestrator.
Tests orchestration logic, service coordination, and workflow management.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

from src.genesis_humanoid_rl.application.services.training_orchestrator import TrainingOrchestrator
from src.genesis_humanoid_rl.domain.model.value_objects import (
    SessionId, RobotId, PlanId, SkillType, MasteryLevel, PerformanceMetrics,
    MotionCommand, MotionType
)
from src.genesis_humanoid_rl.domain.model.entities import (
    LearningEpisode, EpisodeStatus, EpisodeOutcome, CurriculumStage, StageType
)
from src.genesis_humanoid_rl.domain.model.aggregates import (
    LearningSession, SessionStatus, HumanoidRobot, RobotType, CurriculumPlan, PlanStatus
)
from src.genesis_humanoid_rl.domain.services.curriculum_service import AdvancementDecision
from src.genesis_humanoid_rl.application.commands import (
    StartTrainingSessionCommand, ExecuteEpisodeCommand, AdvanceCurriculumCommand
)


class TestTrainingOrchestrator:
    """Test TrainingOrchestrator application service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock dependencies
        self.mock_genesis_adapter = Mock()
        self.mock_movement_analyzer = Mock()
        self.mock_curriculum_service = Mock()
        self.mock_session_repository = Mock()
        self.mock_robot_repository = Mock()
        self.mock_plan_repository = Mock()
        self.mock_event_publisher = Mock()
        
        # Create orchestrator under test
        self.orchestrator = TrainingOrchestrator(
            genesis_adapter=self.mock_genesis_adapter,
            movement_analyzer=self.mock_movement_analyzer,
            curriculum_service=self.mock_curriculum_service,
            session_repository=self.mock_session_repository,
            robot_repository=self.mock_robot_repository,
            plan_repository=self.mock_plan_repository,
            event_publisher=self.mock_event_publisher
        )
        
        # Create test entities
        self.session_id = SessionId.generate()
        self.robot_id = RobotId.generate()
        self.plan_id = PlanId.generate()
        
        self.test_robot = HumanoidRobot(
            robot_id=self.robot_id,
            robot_type=RobotType.UNITREE_G1,
            name="Test Robot",
            joint_count=35,
            height=1.2,
            weight=35.0
        )
        
        self.test_session = LearningSession(
            session_id=self.session_id,
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session",
            max_episodes=100
        )
        
        self.test_plan = CurriculumPlan(
            plan_id=self.plan_id,
            name="Test Curriculum",
            robot_type=RobotType.UNITREE_G1,
            description="Test curriculum plan"
        )
        
        # Set up repository responses
        self.mock_robot_repository.get_by_id.return_value = self.test_robot
        self.mock_session_repository.get_by_id.return_value = self.test_session
        self.mock_plan_repository.get_by_id.return_value = self.test_plan
    
    def test_start_training_session_successful(self):
        """Test successful training session start."""
        command = StartTrainingSessionCommand(
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Integration Test Session",
            max_episodes=50
        )
        
        # Mock successful session creation
        new_session = LearningSession(
            session_id=SessionId.generate(),
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name=command.session_name,
            max_episodes=command.max_episodes
        )
        self.mock_session_repository.save.return_value = new_session
        
        # Mock curriculum plan with stages
        stage = CurriculumStage(
            stage_id="test_stage",
            name="Test Stage",
            stage_type=StageType.FOUNDATION,
            order=0
        )
        self.test_plan.stages = [stage]
        self.test_plan.status = PlanStatus.ACTIVE
        
        result = self.orchestrator.start_training_session(command)
        
        assert result['success'] is True
        assert 'session_id' in result
        assert result['status'] == 'started'
        
        # Verify repositories were called
        self.mock_robot_repository.get_by_id.assert_called_once_with(self.robot_id)
        self.mock_plan_repository.get_by_id.assert_called_once_with(self.plan_id)
        self.mock_session_repository.save.assert_called_once()
        
        # Verify event was published
        self.mock_event_publisher.publish.assert_called()
    
    def test_start_training_session_robot_not_found(self):
        """Test training session start when robot not found."""
        command = StartTrainingSessionCommand(
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session"
        )
        
        # Mock robot not found
        self.mock_robot_repository.get_by_id.return_value = None
        
        result = self.orchestrator.start_training_session(command)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Robot not found' in result['error']
        
        # Should not save session or publish events
        self.mock_session_repository.save.assert_not_called()
        self.mock_event_publisher.publish.assert_not_called()
    
    def test_start_training_session_plan_not_active(self):
        """Test training session start when curriculum plan is not active."""
        command = StartTrainingSessionCommand(
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session"
        )
        
        # Plan is in draft status
        self.test_plan.status = PlanStatus.DRAFT
        
        result = self.orchestrator.start_training_session(command)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'not active' in result['error']
    
    def test_execute_episode_successful(self):
        """Test successful episode execution."""
        command = ExecuteEpisodeCommand(
            session_id=self.session_id,
            target_skill=SkillType.FORWARD_WALKING,
            max_steps=100
        )
        
        # Set up session as active
        self.test_session.status = SessionStatus.ACTIVE
        
        # Mock Genesis simulation results
        self.mock_genesis_adapter.execute_motion_command.return_value = {
            'success': True,
            'robot_state': Mock(),
            'command_executed': Mock()
        }
        self.mock_genesis_adapter.simulate_episode_step.return_value = {
            'success': True,
            'robot_state': Mock(),
            'physics_stable': True,
            'stability_score': 0.8
        }
        
        # Mock movement analysis
        self.mock_movement_analyzer.assess_movement_quality.return_value = {
            'overall_quality_score': 0.7,
            'gait_analysis': Mock(),
            'stability_analysis': Mock()
        }
        
        result = self.orchestrator.execute_episode(command)
        
        assert result['success'] is True
        assert 'episode_id' in result
        assert 'episode_outcome' in result
        assert 'performance_metrics' in result
        
        # Verify session was updated
        self.mock_session_repository.save.assert_called()
        
        # Verify movement analysis was performed
        self.mock_movement_analyzer.assess_movement_quality.assert_called()
    
    def test_execute_episode_session_not_active(self):
        """Test episode execution when session is not active."""
        command = ExecuteEpisodeCommand(
            session_id=self.session_id,
            target_skill=SkillType.FORWARD_WALKING
        )
        
        # Session is not active
        self.test_session.status = SessionStatus.CREATED
        
        result = self.orchestrator.execute_episode(command)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'not active' in result['error']
    
    def test_execute_episode_physics_instability(self):
        """Test episode execution when physics becomes unstable."""
        command = ExecuteEpisodeCommand(
            session_id=self.session_id,
            target_skill=SkillType.FORWARD_WALKING,
            max_steps=100
        )
        
        self.test_session.status = SessionStatus.ACTIVE
        
        # Mock physics instability
        self.mock_genesis_adapter.simulate_episode_step.return_value = {
            'success': True,
            'robot_state': Mock(),
            'physics_stable': False,
            'stability_score': 0.2
        }
        
        result = self.orchestrator.execute_episode(command)
        
        # Episode should terminate early due to instability
        assert result['success'] is True
        assert result['episode_outcome'] == EpisodeOutcome.TERMINATED_EARLY.value
        assert 'physics instability' in result.get('termination_reason', '')
    
    def test_execute_episode_genesis_failure(self):
        """Test episode execution when Genesis fails."""
        command = ExecuteEpisodeCommand(
            session_id=self.session_id,
            target_skill=SkillType.FORWARD_WALKING
        )
        
        self.test_session.status = SessionStatus.ACTIVE
        
        # Mock Genesis failure
        self.mock_genesis_adapter.execute_motion_command.return_value = {
            'success': False,
            'error': 'Genesis simulation error'
        }
        
        result = self.orchestrator.execute_episode(command)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Genesis simulation error' in result['error']
    
    def test_advance_curriculum_ready_for_advancement(self):
        """Test curriculum advancement when robot is ready."""
        command = AdvanceCurriculumCommand(session_id=self.session_id)
        
        # Set up session with episodes
        self.test_session.status = SessionStatus.ACTIVE
        self.test_session.current_stage_index = 0
        
        # Mock curriculum service decision
        advancement_decision = AdvancementDecision(
            should_advance=True,
            confidence_score=0.9,
            success_criteria_met=['episode_count', 'success_rate', 'skill_mastery'],
            remaining_requirements=[]
        )
        self.mock_curriculum_service.evaluate_advancement_readiness.return_value = advancement_decision
        
        # Set up curriculum plan with multiple stages
        stage1 = CurriculumStage(
            stage_id="stage1",
            name="Foundation",
            stage_type=StageType.FOUNDATION,
            order=0
        )
        stage2 = CurriculumStage(
            stage_id="stage2",
            name="Walking",
            stage_type=StageType.SKILL_BUILDING,
            order=1
        )
        self.test_plan.stages = [stage1, stage2]
        
        result = self.orchestrator.advance_curriculum(command)
        
        assert result['success'] is True
        assert result['advanced'] is True
        assert result['new_stage_index'] == 1
        assert result['confidence_score'] == 0.9
        
        # Verify curriculum service was called
        self.mock_curriculum_service.evaluate_advancement_readiness.assert_called()
        
        # Verify session was updated
        self.mock_session_repository.save.assert_called()
        
        # Verify advancement event was published
        self.mock_event_publisher.publish.assert_called()
    
    def test_advance_curriculum_not_ready(self):
        """Test curriculum advancement when robot is not ready."""
        command = AdvanceCurriculumCommand(session_id=self.session_id)
        
        self.test_session.status = SessionStatus.ACTIVE
        
        # Mock curriculum service decision - not ready
        advancement_decision = AdvancementDecision(
            should_advance=False,
            confidence_score=0.4,
            success_criteria_met=['episode_count'],
            remaining_requirements=['success_rate', 'skill_mastery']
        )
        self.mock_curriculum_service.evaluate_advancement_readiness.return_value = advancement_decision
        
        result = self.orchestrator.advance_curriculum(command)
        
        assert result['success'] is True
        assert result['advanced'] is False
        assert result['remaining_requirements'] == ['success_rate', 'skill_mastery']
        assert result['confidence_score'] == 0.4
    
    def test_advance_curriculum_final_stage(self):
        """Test curriculum advancement when already at final stage."""
        command = AdvanceCurriculumCommand(session_id=self.session_id)
        
        # Set session to final stage
        self.test_session.status = SessionStatus.ACTIVE
        self.test_session.current_stage_index = 0
        
        # Only one stage in plan
        stage = CurriculumStage(
            stage_id="final_stage",
            name="Final Stage",
            stage_type=StageType.MASTERY,
            order=0
        )
        self.test_plan.stages = [stage]
        
        result = self.orchestrator.advance_curriculum(command)
        
        assert result['success'] is True
        assert result['advanced'] is False
        assert 'final stage' in result['message']
    
    def test_get_training_progress(self):
        """Test training progress retrieval."""
        # Set up session with some progress
        self.test_session.total_episodes = 25
        self.test_session.successful_episodes = 18
        self.test_session.current_stage_index = 1
        
        # Mock session statistics
        mock_stats = {
            'session_id': self.session_id.value,
            'total_episodes': 25,
            'successful_episodes': 18,
            'success_rate': 0.72,
            'current_stage': 1
        }
        
        with patch.object(self.test_session, 'get_session_statistics', return_value=mock_stats):
            progress = self.orchestrator.get_training_progress(self.session_id)
        
        assert progress['session_id'] == self.session_id.value
        assert progress['total_episodes'] == 25
        assert progress['successful_episodes'] == 18
        assert progress['success_rate'] == 0.72
        assert progress['current_stage'] == 1
    
    def test_get_training_progress_session_not_found(self):
        """Test training progress retrieval when session not found."""
        # Mock session not found
        self.mock_session_repository.get_by_id.return_value = None
        
        progress = self.orchestrator.get_training_progress(self.session_id)
        
        assert 'error' in progress
        assert 'Session not found' in progress['error']
    
    def test_get_robot_capabilities(self):
        """Test robot capabilities assessment."""
        # Mock Genesis adapter capabilities
        mock_capabilities = {
            'balance_capability': 0.8,
            'locomotion_capability': 0.7,
            'stability_rating': 'good',
            'joint_count': 35
        }
        self.mock_genesis_adapter.assess_robot_capabilities.return_value = mock_capabilities
        
        # Mock robot capabilities
        mock_robot_capabilities = {
            'robot_id': self.robot_id.value,
            'mastered_skills': [SkillType.STATIC_BALANCE.value],
            'skill_count': 1,
            'avg_performance': 0.75
        }
        
        with patch.object(self.test_robot, 'get_robot_capabilities', return_value=mock_robot_capabilities):
            capabilities = self.orchestrator.get_robot_capabilities(self.robot_id)
        
        assert capabilities['robot_id'] == self.robot_id.value
        assert capabilities['balance_capability'] == 0.8
        assert capabilities['locomotion_capability'] == 0.7
        assert capabilities['stability_rating'] == 'good'
        assert SkillType.STATIC_BALANCE.value in capabilities['mastered_skills']
    
    def test_error_handling_repository_failure(self):
        """Test error handling when repository operations fail."""
        command = StartTrainingSessionCommand(
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session"
        )
        
        # Mock repository failure
        self.mock_session_repository.save.side_effect = RuntimeError("Database connection failed")
        
        result = self.orchestrator.start_training_session(command)
        
        assert result['success'] is False
        assert 'error' in result
        assert 'Database connection failed' in result['error']
    
    def test_event_publishing_on_major_actions(self):
        """Test that events are published for major orchestration actions."""
        # Test session start
        start_command = StartTrainingSessionCommand(
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session"
        )
        
        # Mock successful setup
        new_session = LearningSession(
            session_id=SessionId.generate(),
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session"
        )
        self.mock_session_repository.save.return_value = new_session
        self.test_plan.status = PlanStatus.ACTIVE
        self.test_plan.stages = [Mock()]
        
        self.orchestrator.start_training_session(start_command)
        
        # Should publish session started event
        assert self.mock_event_publisher.publish.call_count >= 1
        
        # Reset mock
        self.mock_event_publisher.reset_mock()
        
        # Test curriculum advancement
        advance_command = AdvanceCurriculumCommand(session_id=self.session_id)
        self.test_session.status = SessionStatus.ACTIVE
        
        advancement_decision = AdvancementDecision(
            should_advance=True,
            confidence_score=0.9,
            success_criteria_met=[],
            remaining_requirements=[]
        )
        self.mock_curriculum_service.evaluate_advancement_readiness.return_value = advancement_decision
        
        # Set up multiple stages
        self.test_plan.stages = [Mock(), Mock()]
        
        self.orchestrator.advance_curriculum(advance_command)
        
        # Should publish advancement event
        assert self.mock_event_publisher.publish.call_count >= 1
    
    def test_orchestrator_state_consistency(self):
        """Test that orchestrator maintains consistent state across operations."""
        # Start session
        start_command = StartTrainingSessionCommand(
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Consistency Test"
        )
        
        new_session = LearningSession(
            session_id=SessionId.generate(),
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Consistency Test"
        )
        self.mock_session_repository.save.return_value = new_session
        self.test_plan.status = PlanStatus.ACTIVE
        self.test_plan.stages = [Mock()]
        
        start_result = self.orchestrator.start_training_session(start_command)
        assert start_result['success'] is True
        
        # Execute episode
        session_id = SessionId.from_string(start_result['session_id'])
        self.mock_session_repository.get_by_id.return_value = new_session
        new_session.status = SessionStatus.ACTIVE
        
        episode_command = ExecuteEpisodeCommand(
            session_id=session_id,
            target_skill=SkillType.FORWARD_WALKING
        )
        
        # Mock successful episode execution
        self.mock_genesis_adapter.execute_motion_command.return_value = {'success': True, 'robot_state': Mock()}
        self.mock_genesis_adapter.simulate_episode_step.return_value = {
            'success': True, 'robot_state': Mock(), 'physics_stable': True
        }
        self.mock_movement_analyzer.assess_movement_quality.return_value = {'overall_quality_score': 0.7}
        
        episode_result = self.orchestrator.execute_episode(episode_command)
        assert episode_result['success'] is True
        
        # Verify session was saved after each operation
        assert self.mock_session_repository.save.call_count >= 2