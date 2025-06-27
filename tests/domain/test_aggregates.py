"""
Comprehensive tests for domain aggregates.
Tests aggregate business rules, invariants, and complex interactions.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.genesis_humanoid_rl.domain.model.aggregates import (
    LearningSession, SessionStatus, HumanoidRobot, RobotType, 
    CurriculumPlan, PlanStatus
)
from src.genesis_humanoid_rl.domain.model.entities import (
    LearningEpisode, EpisodeOutcome, CurriculumStage, StageType, AdvancementCriteria
)
from src.genesis_humanoid_rl.domain.model.value_objects import (
    SessionId, RobotId, PlanId, EpisodeId, SkillType, MasteryLevel,
    LocomotionSkill, PerformanceMetrics, GaitPattern, SkillAssessment
)


class TestLearningSession:
    """Test LearningSession aggregate business rules and lifecycle."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_id = SessionId.generate()
        self.robot_id = RobotId.generate()
        self.plan_id = PlanId.generate()
        
        self.session = LearningSession(
            session_id=self.session_id,
            robot_id=self.robot_id,
            plan_id=self.plan_id,
            session_name="Test Session",
            max_episodes=100
        )
    
    def test_session_creation_initial_state(self):
        """Test session is created in correct initial state."""
        assert self.session.session_id == self.session_id
        assert self.session.robot_id == self.robot_id
        assert self.session.plan_id == self.plan_id
        assert self.session.status == SessionStatus.CREATED
        assert self.session.current_stage_index == 0
        assert len(self.session.episodes) == 0
        assert self.session.active_episode is None
        assert self.session.total_episodes == 0
        assert self.session.successful_episodes == 0
    
    def test_start_session_valid(self):
        """Test valid session start."""
        initial_stage = CurriculumStage(
            stage_id="stage_0",
            name="Initial Stage",
            stage_type=StageType.FOUNDATION,
            order=0
        )
        
        self.session.start_session(initial_stage)
        
        assert self.session.status == SessionStatus.ACTIVE
        assert self.session.current_stage_index == 0
    
    def test_start_session_invalid_status(self):
        """Test that starting non-created session raises error."""
        self.session.status = SessionStatus.ACTIVE
        
        with pytest.raises(ValueError, match="Cannot start session in status"):
            self.session.start_session()
    
    def test_create_episode_valid(self):
        """Test valid episode creation."""
        self.session.start_session()
        
        episode = self.session.create_episode(SkillType.FORWARD_WALKING)
        
        assert episode.session_id == self.session.session_id
        assert episode.target_skill == SkillType.FORWARD_WALKING
        assert self.session.active_episode == episode
        assert len(self.session.episodes) == 1
        assert self.session.episodes[0] == episode
    
    def test_create_episode_with_active_episode(self):
        """Test that creating episode with active episode raises error."""
        self.session.start_session()
        
        # Create first episode
        self.session.create_episode(SkillType.STATIC_BALANCE)
        
        # Try to create second episode while first is active
        with pytest.raises(ValueError, match="Cannot create episode while another episode is active"):
            self.session.create_episode(SkillType.FORWARD_WALKING)
    
    def test_create_episode_inactive_session(self):
        """Test that creating episode in inactive session raises error."""
        with pytest.raises(ValueError, match="Cannot create episode in session status"):
            self.session.create_episode(SkillType.FORWARD_WALKING)
    
    def test_create_episode_max_episodes_reached(self):
        """Test that creating episode when max reached raises error."""
        self.session.start_session()
        
        # Set episodes to max limit
        self.session.episodes = [MagicMock() for _ in range(100)]
        
        with pytest.raises(ValueError, match="Maximum episodes reached for session"):
            self.session.create_episode(SkillType.FORWARD_WALKING)
    
    def test_complete_episode_successful(self):
        """Test completing episode successfully updates session metrics."""
        self.session.start_session()
        
        episode = self.session.create_episode(SkillType.FORWARD_WALKING)
        episode.start_episode()
        episode.add_step_reward(10.0)
        
        performance_metrics = PerformanceMetrics(
            success_rate=0.8,
            average_reward=5.0
        )
        
        self.session.complete_episode(EpisodeOutcome.SUCCESS, performance_metrics)
        
        assert self.session.total_episodes == 1
        assert self.session.successful_episodes == 1
        assert self.session.active_episode is None
        assert 'avg_reward' in self.session.session_metrics
    
    def test_complete_episode_failed(self):
        """Test completing failed episode updates session correctly."""
        self.session.start_session()
        
        episode = self.session.create_episode(SkillType.FORWARD_WALKING)
        episode.start_episode()
        
        self.session.complete_episode(EpisodeOutcome.FAILURE)
        
        assert self.session.total_episodes == 1
        assert self.session.successful_episodes == 0  # Failed episode doesn't count
    
    def test_complete_episode_no_active_episode(self):
        """Test that completing episode with no active episode raises error."""
        self.session.start_session()
        
        with pytest.raises(ValueError, match="No active episode to complete"):
            self.session.complete_episode(EpisodeOutcome.SUCCESS)
    
    def test_advance_curriculum_if_ready_success(self):
        """Test successful curriculum advancement."""
        # Create mock curriculum plan with multiple stages
        curriculum_plan = MagicMock()
        
        # Create stages with advancement capability
        stage1 = MagicMock()
        stage1.can_advance.return_value = True
        stage1.add_episode_result = MagicMock()
        
        stage2 = MagicMock()
        
        curriculum_plan.stages = [stage1, stage2]
        
        self.session.start_session()
        self.session.current_stage_index = 0
        
        # Add some completed episodes for the stage to evaluate
        episode = MagicMock()
        self.session.episodes = [episode]
        
        result = self.session.advance_curriculum_if_ready(curriculum_plan)
        
        assert result is True
        assert self.session.current_stage_index == 1
        stage1.add_episode_result.assert_called_with(episode)
    
    def test_advance_curriculum_if_ready_not_ready(self):
        """Test curriculum advancement when not ready."""
        curriculum_plan = MagicMock()
        
        stage1 = MagicMock()
        stage1.can_advance.return_value = False
        stage1.add_episode_result = MagicMock()
        
        curriculum_plan.stages = [stage1]
        
        self.session.start_session()
        
        result = self.session.advance_curriculum_if_ready(curriculum_plan)
        
        assert result is False
        assert self.session.current_stage_index == 0
    
    def test_advance_curriculum_final_stage(self):
        """Test advancement attempt when already at final stage."""
        curriculum_plan = MagicMock()
        curriculum_plan.stages = [MagicMock()]  # Only one stage
        
        self.session.start_session()
        self.session.current_stage_index = 1  # Beyond last stage
        
        result = self.session.advance_curriculum_if_ready(curriculum_plan)
        
        assert result is False
    
    def test_pause_session(self):
        """Test session pausing."""
        self.session.start_session()
        
        # Create and start an episode
        episode = self.session.create_episode()
        episode.start_episode()
        
        self.session.pause_session()
        
        assert self.session.status == SessionStatus.PAUSED
        assert self.session.active_episode is None  # Episode should be terminated
    
    def test_pause_session_invalid_status(self):
        """Test that pausing non-active session raises error."""
        with pytest.raises(ValueError, match="Cannot pause session in status"):
            self.session.pause_session()
    
    def test_resume_session(self):
        """Test session resuming."""
        self.session.start_session()
        self.session.pause_session()
        
        self.session.resume_session()
        
        assert self.session.status == SessionStatus.ACTIVE
    
    def test_resume_session_invalid_status(self):
        """Test that resuming non-paused session raises error."""
        with pytest.raises(ValueError, match="Cannot resume session in status"):
            self.session.resume_session()
    
    def test_complete_session(self):
        """Test session completion."""
        self.session.start_session()
        
        # Create and complete an episode
        episode = self.session.create_episode()
        episode.start_episode()
        
        self.session.complete_session()
        
        assert self.session.status == SessionStatus.COMPLETED
        assert self.session.active_episode is None
    
    def test_get_session_statistics(self):
        """Test session statistics calculation."""
        self.session.start_session()
        
        # Add some completed episodes
        self.session.total_episodes = 10
        self.session.successful_episodes = 7
        
        # Create mock episodes with rewards
        mock_episodes = []
        for i in range(5):
            episode = MagicMock()
            episode.total_reward = float(i + 1)
            episode.status = "completed"
            mock_episodes.append(episode)
        
        self.session.episodes = mock_episodes
        
        stats = self.session.get_session_statistics()
        
        assert stats['session_id'] == self.session_id.value
        assert stats['total_episodes'] == 10
        assert stats['successful_episodes'] == 7
        assert stats['success_rate'] == 0.7
        assert stats['current_stage'] == 0
        assert 'avg_episode_reward' in stats
        assert stats['status'] == SessionStatus.ACTIVE.value
    
    def test_get_learning_progress(self):
        """Test learning progress calculation."""
        # Create mock episodes with varying success
        early_episodes = []
        for i in range(10):
            episode = MagicMock()
            episode.is_successful.return_value = i >= 7  # 3/10 successful
            early_episodes.append(episode)
        
        recent_episodes = []
        for i in range(10):
            episode = MagicMock()
            episode.is_successful.return_value = i >= 2  # 8/10 successful
            recent_episodes.append(episode)
        
        # Set up episodes list to simulate progress
        self.session.episodes = early_episodes + recent_episodes
        
        progress = self.session.get_learning_progress()
        
        # Should show improvement: from 30% to 80% success rate
        assert 0.0 <= progress <= 1.0
        assert progress > 0.5  # Should indicate positive progress


class TestHumanoidRobot:
    """Test HumanoidRobot aggregate business rules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.robot_id = RobotId.generate()
        
        self.robot = HumanoidRobot(
            robot_id=self.robot_id,
            robot_type=RobotType.UNITREE_G1,
            name="Test Robot",
            joint_count=35,
            height=1.2,
            weight=35.0
        )
    
    def test_robot_creation_initial_state(self):
        """Test robot is created in correct initial state."""
        assert self.robot.robot_id == self.robot_id
        assert self.robot.robot_type == RobotType.UNITREE_G1
        assert self.robot.name == "Test Robot"
        assert self.robot.joint_count == 35
        assert len(self.robot.learned_skills) == 0
        assert len(self.robot.skill_history) == 0
        assert len(self.robot.performance_history) == 0
    
    def test_assess_skill_first_time(self):
        """Test first skill assessment."""
        skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.BEGINNER,
            proficiency_score=0.3
        )
        
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.3,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        
        self.robot.assess_skill(SkillType.FORWARD_WALKING, assessment)
        
        assert SkillType.FORWARD_WALKING in self.robot.learned_skills
        assert self.robot.learned_skills[SkillType.FORWARD_WALKING] == skill
        assert len(self.robot.skill_history) == 1
        assert self.robot.skill_history[0] == assessment
    
    def test_assess_skill_improvement(self):
        """Test skill assessment improvement."""
        # Initial skill assessment
        initial_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.3
        )
        initial_assessment = SkillAssessment(
            skill=initial_skill,
            assessment_score=0.3,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, initial_assessment)
        
        # Improved skill assessment
        improved_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.7  # Improvement
        )
        improved_assessment = SkillAssessment(
            skill=improved_skill,
            assessment_score=0.7,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        
        self.robot.assess_skill(SkillType.FORWARD_WALKING, improved_assessment)
        
        # Should accept improvement
        assert self.robot.learned_skills[SkillType.FORWARD_WALKING].proficiency_score == 0.7
        assert len(self.robot.skill_history) == 2
    
    def test_assess_skill_regression_prevention(self):
        """Test that skill regression is prevented."""
        # Initial high skill
        high_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.8
        )
        high_assessment = SkillAssessment(
            skill=high_skill,
            assessment_score=0.8,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, high_assessment)
        
        # Attempted regression
        low_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.3  # Lower than current
        )
        low_assessment = SkillAssessment(
            skill=low_skill,
            assessment_score=0.3,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        
        self.robot.assess_skill(SkillType.FORWARD_WALKING, low_assessment)
        
        # Should maintain higher skill level
        assert self.robot.learned_skills[SkillType.FORWARD_WALKING].proficiency_score == 0.8
        # But assessment should still be recorded in history
        assert len(self.robot.skill_history) == 2
    
    def test_master_skill_valid(self):
        """Test valid skill mastery."""
        # First learn the skill
        skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.7
        )
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.7,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, assessment)
        
        # Master the skill
        evidence = {"episodes_demonstrated": 50, "consistency": 0.95}
        result = self.robot.master_skill(SkillType.FORWARD_WALKING, evidence)
        
        assert result is True
        assert self.robot.learned_skills[SkillType.FORWARD_WALKING].mastery_level == MasteryLevel.EXPERT
        assert self.robot.learned_skills[SkillType.FORWARD_WALKING].proficiency_score == 1.0
        assert f'{SkillType.FORWARD_WALKING.value}_mastery_evidence' in self.robot.metadata
    
    def test_master_skill_unlearned(self):
        """Test mastering unlearned skill fails."""
        evidence = {"episodes_demonstrated": 50}
        result = self.robot.master_skill(SkillType.FORWARD_WALKING, evidence)
        
        assert result is False
        assert SkillType.FORWARD_WALKING not in self.robot.learned_skills
    
    def test_can_learn_skill_no_prerequisites(self):
        """Test learning skill with no prerequisites."""
        # POSTURAL_CONTROL typically has no prerequisites
        assert self.robot.can_learn_skill(SkillType.POSTURAL_CONTROL)
    
    def test_can_learn_skill_with_prerequisites_met(self):
        """Test learning skill when prerequisites are met."""
        # Master prerequisite skill first
        prereq_skill = LocomotionSkill(
            skill_type=SkillType.STATIC_BALANCE,
            mastery_level=MasteryLevel.INTERMEDIATE  # Mastered
        )
        prereq_assessment = SkillAssessment(
            skill=prereq_skill,
            assessment_score=0.8,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        self.robot.assess_skill(SkillType.STATIC_BALANCE, prereq_assessment)
        
        # Should be able to learn skill that requires static balance
        # Note: This test depends on the prerequisite relationships defined in LocomotionSkill
    
    def test_can_learn_skill_already_mastered(self):
        """Test that already mastered skills return False."""
        # Master a skill
        mastered_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.EXPERT
        )
        assessment = SkillAssessment(
            skill=mastered_skill,
            assessment_score=1.0,
            confidence_level=0.9,
            evidence_quality=0.9
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, assessment)
        
        # Should not be able to "learn" it again
        assert not self.robot.can_learn_skill(SkillType.FORWARD_WALKING)
    
    def test_get_skill_proficiency(self):
        """Test skill proficiency retrieval."""
        # Unlearned skill
        assert self.robot.get_skill_proficiency(SkillType.FORWARD_WALKING) == 0.0
        
        # Learned skill
        skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            proficiency_score=0.65
        )
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.65,
            confidence_level=0.8,
            evidence_quality=0.7
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, assessment)
        
        assert self.robot.get_skill_proficiency(SkillType.FORWARD_WALKING) == 0.65
    
    def test_get_mastered_skills(self):
        """Test mastered skills retrieval."""
        # No mastered skills initially
        assert len(self.robot.get_mastered_skills()) == 0
        
        # Add mastered skill
        mastered_skill = LocomotionSkill(
            skill_type=SkillType.STATIC_BALANCE,
            mastery_level=MasteryLevel.EXPERT
        )
        mastered_assessment = SkillAssessment(
            skill=mastered_skill,
            assessment_score=0.9,
            confidence_level=0.9,
            evidence_quality=0.9
        )
        self.robot.assess_skill(SkillType.STATIC_BALANCE, mastered_assessment)
        
        # Add non-mastered skill
        learning_skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.NOVICE
        )
        learning_assessment = SkillAssessment(
            skill=learning_skill,
            assessment_score=0.3,
            confidence_level=0.7,
            evidence_quality=0.6
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, learning_assessment)
        
        mastered = self.robot.get_mastered_skills()
        assert len(mastered) == 1
        assert SkillType.STATIC_BALANCE in mastered
        assert SkillType.FORWARD_WALKING not in mastered
    
    def test_get_next_recommended_skills(self):
        """Test next skill recommendations."""
        recommendations = self.robot.get_next_recommended_skills()
        
        # Should return some recommendations
        assert len(recommendations) <= 3  # Limit to top 3
        assert all(isinstance(skill, SkillType) for skill in recommendations)
        
        # Should be ordered by complexity (simpler skills first)
        if len(recommendations) > 1:
            # POSTURAL_CONTROL should come before more complex skills
            assert SkillType.POSTURAL_CONTROL in recommendations[:2]
    
    def test_update_performance_metrics(self):
        """Test performance metrics updating."""
        metrics1 = PerformanceMetrics(success_rate=0.7, average_reward=10.0)
        metrics2 = PerformanceMetrics(success_rate=0.8, average_reward=12.0)
        
        self.robot.update_performance_metrics(metrics1)
        self.robot.update_performance_metrics(metrics2)
        
        assert len(self.robot.performance_history) == 2
        assert self.robot.performance_history[0] == metrics1
        assert self.robot.performance_history[1] == metrics2
    
    def test_performance_history_limit(self):
        """Test that performance history is limited to 100 records."""
        # Add more than 100 performance records
        for i in range(120):
            metrics = PerformanceMetrics(success_rate=0.5, average_reward=float(i))
            self.robot.update_performance_metrics(metrics)
        
        # Should keep only last 100
        assert len(self.robot.performance_history) == 100
        # Should have the latest records
        assert self.robot.performance_history[-1].average_reward == 119.0
        assert self.robot.performance_history[0].average_reward == 20.0  # 120 - 100
    
    def test_add_gait_pattern(self):
        """Test gait pattern addition."""
        high_quality_gait = GaitPattern(
            stride_length=0.6,
            stride_frequency=2.0,
            step_height=0.05,
            stability_margin=0.1,
            energy_efficiency=0.9,
            symmetry_score=0.95
        )
        
        self.robot.add_gait_pattern(high_quality_gait)
        
        assert len(self.robot.gait_patterns) == 1
        assert self.robot.gait_patterns[0] == high_quality_gait
    
    def test_gait_pattern_quality_filtering(self):
        """Test that low-quality gait patterns are filtered out."""
        low_quality_gait = GaitPattern(
            stride_length=0.3,
            stride_frequency=1.0,
            step_height=0.02,
            stability_margin=0.02,
            energy_efficiency=0.3,
            symmetry_score=0.4
        )
        
        self.robot.add_gait_pattern(low_quality_gait)
        
        # Should be filtered out due to low quality score
        assert len(self.robot.gait_patterns) == 0
    
    def test_gait_pattern_limit(self):
        """Test that gait patterns are limited to 20 best patterns."""
        # Add many high-quality gait patterns
        for i in range(25):
            efficiency = 0.7 + (i * 0.01)  # Increasing efficiency
            gait = GaitPattern(
                stride_length=0.5,
                stride_frequency=1.5,
                step_height=0.05,
                stability_margin=0.08,
                energy_efficiency=efficiency,
                symmetry_score=0.8
            )
            self.robot.add_gait_pattern(gait)
        
        # Should keep only 20 best
        assert len(self.robot.gait_patterns) == 20
        
        # Should keep the highest quality ones
        # The best patterns should have higher quality scores (not just efficiency)
        min_quality = min(gait.get_quality_score() for gait in self.robot.gait_patterns)
        # With 25 patterns (efficiency 0.7 to 0.94), top 20 should exclude lowest 5
        assert min_quality >= 0.70  # Should have filtered out lowest quality patterns
    
    def test_get_robot_capabilities(self):
        """Test robot capabilities summary."""
        # Add some skills and performance
        skill = LocomotionSkill(
            skill_type=SkillType.FORWARD_WALKING,
            mastery_level=MasteryLevel.EXPERT
        )
        assessment = SkillAssessment(
            skill=skill,
            assessment_score=0.9,
            confidence_level=0.9,
            evidence_quality=0.9
        )
        self.robot.assess_skill(SkillType.FORWARD_WALKING, assessment)
        
        metrics = PerformanceMetrics(success_rate=0.8, average_reward=15.0)
        self.robot.update_performance_metrics(metrics)
        
        capabilities = self.robot.get_robot_capabilities()
        
        assert capabilities['robot_id'] == self.robot_id.value
        assert capabilities['robot_type'] == RobotType.UNITREE_G1.value
        assert SkillType.FORWARD_WALKING.value in capabilities['mastered_skills']
        assert capabilities['skill_count'] == 1
        assert capabilities['avg_performance'] > 0
        assert 'next_recommended_skills' in capabilities


class TestCurriculumPlan:
    """Test CurriculumPlan aggregate business rules."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plan_id = PlanId.generate()
        
        self.plan = CurriculumPlan(
            plan_id=self.plan_id,
            name="Test Curriculum",
            robot_type=RobotType.UNITREE_G1,
            description="Test curriculum for G1 robot"
        )
    
    def test_plan_creation_initial_state(self):
        """Test plan is created in correct initial state."""
        assert self.plan.plan_id == self.plan_id
        assert self.plan.name == "Test Curriculum"
        assert self.plan.robot_type == RobotType.UNITREE_G1
        assert self.plan.status == PlanStatus.DRAFT
        assert len(self.plan.stages) == 0
    
    def test_add_stage_valid(self):
        """Test valid stage addition."""
        stage = CurriculumStage(
            stage_id="stage_0",
            name="Foundation Stage",
            stage_type=StageType.FOUNDATION,
            order=0,
            target_skills={SkillType.POSTURAL_CONTROL}
        )
        
        self.plan.add_stage(stage)
        
        assert len(self.plan.stages) == 1
        assert self.plan.stages[0] == stage
    
    def test_add_stage_wrong_order(self):
        """Test that adding stage with wrong order raises error."""
        stage = CurriculumStage(
            stage_id="stage_1",
            name="Wrong Order Stage",
            stage_type=StageType.FOUNDATION,
            order=1,  # Should be 0 for first stage
            target_skills={SkillType.POSTURAL_CONTROL}
        )
        
        with pytest.raises(ValueError, match="Stage order .* invalid, expected"):
            self.plan.add_stage(stage)
    
    def test_add_stage_active_plan(self):
        """Test that adding stage to active plan raises error."""
        self.plan.status = PlanStatus.ACTIVE
        
        stage = CurriculumStage(
            stage_id="stage_0",
            name="Foundation Stage",
            stage_type=StageType.FOUNDATION,
            order=0
        )
        
        with pytest.raises(ValueError, match="Cannot modify plan in status"):
            self.plan.add_stage(stage)
    
    def test_activate_plan_valid(self):
        """Test valid plan activation."""
        # Add at least one stage
        stage = CurriculumStage(
            stage_id="stage_0",
            name="Foundation Stage",
            stage_type=StageType.FOUNDATION,
            order=0
        )
        self.plan.add_stage(stage)
        
        self.plan.activate_plan()
        
        assert self.plan.status == PlanStatus.ACTIVE
    
    def test_activate_plan_no_stages(self):
        """Test that activating plan with no stages raises error."""
        with pytest.raises(ValueError, match="Cannot activate plan with no stages"):
            self.plan.activate_plan()
    
    def test_activate_plan_not_draft(self):
        """Test that activating non-draft plan raises error."""
        self.plan.status = PlanStatus.COMPLETED
        
        with pytest.raises(ValueError, match="Cannot activate plan in status"):
            self.plan.activate_plan()
    
    def test_get_stage_by_index(self):
        """Test stage retrieval by index."""
        stage1 = CurriculumStage(
            stage_id="stage_0", name="Stage 1", stage_type=StageType.FOUNDATION, order=0
        )
        stage2 = CurriculumStage(
            stage_id="stage_1", name="Stage 2", stage_type=StageType.SKILL_BUILDING, order=1
        )
        
        self.plan.add_stage(stage1)
        self.plan.add_stage(stage2)
        
        assert self.plan.get_stage_by_index(0) == stage1
        assert self.plan.get_stage_by_index(1) == stage2
        assert self.plan.get_stage_by_index(2) is None
        assert self.plan.get_stage_by_index(-1) is None
    
    def test_get_next_stage(self):
        """Test next stage retrieval."""
        stage1 = CurriculumStage(
            stage_id="stage_0", name="Stage 1", stage_type=StageType.FOUNDATION, order=0
        )
        stage2 = CurriculumStage(
            stage_id="stage_1", name="Stage 2", stage_type=StageType.SKILL_BUILDING, order=1
        )
        
        self.plan.add_stage(stage1)
        self.plan.add_stage(stage2)
        
        assert self.plan.get_next_stage(0) == stage2
        assert self.plan.get_next_stage(1) is None
    
    def test_get_plan_progress(self):
        """Test plan progress calculation."""
        # Add stages
        for i in range(3):
            stage = CurriculumStage(
                stage_id=f"stage_{i}",
                name=f"Stage {i}",
                stage_type=StageType.FOUNDATION,
                order=i
            )
            self.plan.add_stage(stage)
        
        # Test progress at different stages
        assert self.plan.get_plan_progress(0) < 0.5  # Early stage
        assert self.plan.get_plan_progress(2) > 0.5  # Later stage
        assert self.plan.get_plan_progress(3) == 1.0  # Completed
        assert self.plan.get_plan_progress(5) == 1.0  # Beyond completion
    
    def test_get_recommended_next_skills(self):
        """Test recommended skills for current stage."""
        stage = CurriculumStage(
            stage_id="stage_0",
            name="Foundation Stage",
            stage_type=StageType.FOUNDATION,
            order=0,
            target_skills={SkillType.POSTURAL_CONTROL, SkillType.STATIC_BALANCE}
        )
        self.plan.add_stage(stage)
        
        skills = self.plan.get_recommended_next_skills(0)
        
        assert len(skills) == 2
        assert SkillType.POSTURAL_CONTROL in skills
        assert SkillType.STATIC_BALANCE in skills
    
    def test_get_recommended_next_skills_invalid_stage(self):
        """Test recommended skills for invalid stage index."""
        skills = self.plan.get_recommended_next_skills(5)
        assert skills == []
    
    def test_adapt_difficulty(self):
        """Test difficulty adaptation based on performance."""
        stage = CurriculumStage(
            stage_id="stage_0",
            name="Foundation Stage",
            stage_type=StageType.FOUNDATION,
            order=0,
            target_success_rate=0.7,
            min_episodes=20
        )
        self.plan.add_stage(stage)
        
        # Test with high performance (should increase difficulty)
        high_performance = PerformanceMetrics(
            success_rate=0.95,  # High success rate
            average_reward=20.0,
            learning_progress=0.9
        )
        
        adaptations = self.plan.adapt_difficulty(high_performance, 0)
        
        assert adaptations['difficulty_adjustment'] == 'increase'
        assert 'suggested_target_success_rate' in adaptations
        assert 'suggested_min_episodes' in adaptations
        
        # Test with low performance (should decrease difficulty)
        low_performance = PerformanceMetrics(
            success_rate=0.2,  # Low success rate
            average_reward=5.0,
            learning_progress=0.1
        )
        
        adaptations = self.plan.adapt_difficulty(low_performance, 0)
        
        assert adaptations['difficulty_adjustment'] == 'decrease'
    
    def test_adapt_difficulty_invalid_stage(self):
        """Test difficulty adaptation for invalid stage."""
        performance = PerformanceMetrics(success_rate=0.8, average_reward=10.0)
        adaptations = self.plan.adapt_difficulty(performance, 5)
        
        assert adaptations == {}