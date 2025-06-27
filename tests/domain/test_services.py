"""
Comprehensive tests for domain services.
Tests complex business logic, algorithms, and service interactions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.genesis_humanoid_rl.domain.services.movement_analyzer import (
    MovementQualityAnalyzer, GaitAnalysisResult, StabilityAnalysisResult
)
from src.genesis_humanoid_rl.domain.services.curriculum_service import (
    CurriculumProgressionService, AdvancementDecision, DifficultyAdjustment
)
from src.genesis_humanoid_rl.domain.model.value_objects import (
    MovementTrajectory, GaitPattern, PerformanceMetrics, SkillType, MasteryLevel,
    LocomotionSkill, SkillAssessment
)
from src.genesis_humanoid_rl.domain.model.entities import (
    LearningEpisode, EpisodeStatus, EpisodeOutcome, CurriculumStage, StageType
)
from src.genesis_humanoid_rl.domain.model.aggregates import (
    HumanoidRobot, RobotType, LearningSession, CurriculumPlan
)


class TestMovementQualityAnalyzer:
    """Test MovementQualityAnalyzer service business logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MovementQualityAnalyzer()
        
        # Create test trajectory
        self.smooth_trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (2.0, 0.0, 0.8), (3.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0, 2.0, 3.0],
            velocities=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
        )
        
        self.jerky_trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (0.1, 0.0, 0.8), (2.0, 0.0, 0.8), (2.1, 0.0, 0.8)],
            timestamps=[0.0, 1.0, 2.0, 3.0],
            velocities=[(0.1, 0.0, 0.0), (1.9, 0.0, 0.0), (0.1, 0.0, 0.0), (1.9, 0.0, 0.0)]
        )
    
    def test_analyze_gait_stability_smooth_trajectory(self):
        """Test gait stability analysis with smooth trajectory."""
        result = self.analyzer.analyze_gait_stability(self.smooth_trajectory)
        
        assert isinstance(result, StabilityAnalysisResult)
        assert result.overall_score >= 0.7  # Should be high for smooth trajectory
        assert result.velocity_consistency > 0.8  # High consistency
        assert result.acceleration_smoothness > 0.7  # High smoothness
        assert result.is_stable()
    
    def test_analyze_gait_stability_jerky_trajectory(self):
        """Test gait stability analysis with jerky trajectory."""
        result = self.analyzer.analyze_gait_stability(self.jerky_trajectory)
        
        assert isinstance(result, StabilityAnalysisResult)
        assert result.overall_score < 0.5  # Should be low for jerky trajectory
        assert result.velocity_consistency < 0.3  # Low consistency
        assert result.acceleration_smoothness < 0.3  # Low smoothness
        assert not result.is_stable()
    
    def test_analyze_gait_stability_insufficient_data(self):
        """Test gait stability analysis with insufficient trajectory data."""
        short_trajectory = MovementTrajectory(
            positions=[(0.0, 0.0, 0.8), (1.0, 0.0, 0.8)],
            timestamps=[0.0, 1.0]
        )
        
        result = self.analyzer.analyze_gait_stability(short_trajectory)
        
        assert result.overall_score == 0.0
        assert not result.is_stable()
        assert "Insufficient data" in result.analysis_notes
    
    def test_extract_gait_pattern_from_trajectory(self):
        """Test gait pattern extraction from trajectory."""
        gait_pattern = self.analyzer.extract_gait_pattern_from_trajectory(self.smooth_trajectory)
        
        assert isinstance(gait_pattern, GaitPattern)
        assert gait_pattern.stride_length > 0
        assert gait_pattern.stride_frequency > 0
        assert gait_pattern.step_height >= 0
        assert 0.0 <= gait_pattern.energy_efficiency <= 1.0
        assert 0.0 <= gait_pattern.symmetry_score <= 1.0
    
    def test_extract_gait_pattern_quality_correlation(self):
        """Test that gait pattern quality correlates with trajectory smoothness."""
        smooth_gait = self.analyzer.extract_gait_pattern_from_trajectory(self.smooth_trajectory)
        jerky_gait = self.analyzer.extract_gait_pattern_from_trajectory(self.jerky_trajectory)
        
        smooth_quality = smooth_gait.get_quality_score()
        jerky_quality = jerky_gait.get_quality_score()
        
        assert smooth_quality > jerky_quality
    
    def test_assess_movement_quality_comprehensive(self):
        """Test comprehensive movement quality assessment."""
        # Create test episode with trajectory and performance
        episode = Mock()
        episode.movement_trajectory = self.smooth_trajectory
        episode.performance_metrics = PerformanceMetrics(
            success_rate=0.8,
            average_reward=10.0,
            skill_scores={SkillType.FORWARD_WALKING: 0.7}
        )
        episode.get_duration.return_value = timedelta(minutes=2)
        episode.step_count = 120
        
        quality_assessment = self.analyzer.assess_movement_quality(episode)
        
        assert isinstance(quality_assessment, dict)
        assert 'gait_analysis' in quality_assessment
        assert 'stability_analysis' in quality_assessment
        assert 'efficiency_metrics' in quality_assessment
        assert 'overall_quality_score' in quality_assessment
        
        assert 0.0 <= quality_assessment['overall_quality_score'] <= 1.0
    
    def test_compare_gait_patterns(self):
        """Test gait pattern comparison algorithm."""
        # Create similar gait patterns
        gait1 = GaitPattern(
            stride_length=0.6,
            stride_frequency=2.0,
            step_height=0.05,
            stability_margin=0.1,
            energy_efficiency=0.8,
            symmetry_score=0.9
        )
        
        gait2 = GaitPattern(
            stride_length=0.65,  # Slightly different
            stride_frequency=2.1,
            step_height=0.06,
            stability_margin=0.11,
            energy_efficiency=0.82,
            symmetry_score=0.88
        )
        
        gait3 = GaitPattern(
            stride_length=0.3,   # Very different
            stride_frequency=1.0,
            step_height=0.02,
            stability_margin=0.02,
            energy_efficiency=0.3,
            symmetry_score=0.4
        )
        
        similarity_close = self.analyzer.compare_gait_patterns(gait1, gait2)
        similarity_distant = self.analyzer.compare_gait_patterns(gait1, gait3)
        
        assert 0.0 <= similarity_close <= 1.0
        assert 0.0 <= similarity_distant <= 1.0
        assert similarity_close > similarity_distant
        assert similarity_close > 0.8  # Should be very similar
        assert similarity_distant < 0.5  # Should be quite different
    
    def test_identify_movement_anomalies(self):
        """Test movement anomaly detection."""
        # Create trajectory with anomaly
        anomalous_trajectory = MovementTrajectory(
            positions=[
                (0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (1.0, 0.0, 2.0),  # Sudden jump
                (2.0, 0.0, 0.8), (3.0, 0.0, 0.8)
            ],
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            velocities=[
                (1.0, 0.0, 0.0), (0.0, 0.0, 1.2), (1.0, 0.0, -1.2),  # Velocity spike
                (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
            ]
        )
        
        anomalies = self.analyzer.identify_movement_anomalies(anomalous_trajectory)
        
        assert isinstance(anomalies, list)
        assert len(anomalies) > 0
        
        # Check for height anomaly
        height_anomalies = [a for a in anomalies if a['type'] == 'height_spike']
        assert len(height_anomalies) > 0
        
        # Check for velocity anomaly
        velocity_anomalies = [a for a in anomalies if a['type'] == 'velocity_spike']
        assert len(velocity_anomalies) > 0
    
    def test_calculate_energy_efficiency(self):
        """Test energy efficiency calculation."""
        # Test efficient trajectory (smooth, constant velocity)
        efficient_trajectory = self.smooth_trajectory
        
        # Test inefficient trajectory (many direction changes)
        inefficient_trajectory = MovementTrajectory(
            positions=[
                (0.0, 0.0, 0.8), (1.0, 0.0, 0.8), (0.0, 0.0, 0.8),
                (1.0, 0.0, 0.8), (0.0, 0.0, 0.8)
            ],
            timestamps=[0.0, 1.0, 2.0, 3.0, 4.0],
            velocities=[
                (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)
            ]
        )
        
        efficient_score = self.analyzer.calculate_energy_efficiency(efficient_trajectory)
        inefficient_score = self.analyzer.calculate_energy_efficiency(inefficient_trajectory)
        
        assert 0.0 <= efficient_score <= 1.0
        assert 0.0 <= inefficient_score <= 1.0
        assert efficient_score > inefficient_score
    
    def test_evaluate_balance_quality(self):
        """Test balance quality evaluation."""
        # Create test robot states representing different balance qualities
        stable_states = [
            {'position': [0.0, 0.0, 0.85], 'orientation': [0.0, 0.0, 0.0, 1.0]},
            {'position': [0.01, 0.01, 0.84], 'orientation': [0.02, 0.01, 0.0, 0.999]},
            {'position': [0.0, -0.01, 0.86], 'orientation': [0.01, 0.02, 0.0, 0.999]}
        ]
        
        unstable_states = [
            {'position': [0.0, 0.0, 0.85], 'orientation': [0.3, 0.2, 0.0, 0.8]},
            {'position': [0.2, 0.1, 0.75], 'orientation': [0.4, 0.3, 0.0, 0.7]},
            {'position': [0.1, -0.3, 0.8], 'orientation': [0.2, 0.4, 0.0, 0.8]}
        ]
        
        stable_quality = self.analyzer.evaluate_balance_quality(stable_states)
        unstable_quality = self.analyzer.evaluate_balance_quality(unstable_states)
        
        assert 0.0 <= stable_quality <= 1.0
        assert 0.0 <= unstable_quality <= 1.0
        assert stable_quality > unstable_quality
        assert stable_quality > 0.7  # Should be high for stable states
        assert unstable_quality < 0.5  # Should be low for unstable states


class TestCurriculumProgressionService:
    """Test CurriculumProgressionService business logic."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = CurriculumProgressionService()
        
        # Create test robot with some skills
        self.robot = Mock()
        self.robot.learned_skills = {
            SkillType.STATIC_BALANCE: LocomotionSkill(
                skill_type=SkillType.STATIC_BALANCE,
                mastery_level=MasteryLevel.INTERMEDIATE,
                proficiency_score=0.6
            )
        }
        self.robot.get_skill_proficiency.side_effect = lambda skill: (
            0.6 if skill == SkillType.STATIC_BALANCE else 0.0
        )
        
        # Create test curriculum stage
        self.stage = Mock()
        self.stage.target_skills = {SkillType.STATIC_BALANCE, SkillType.FORWARD_WALKING}
        self.stage.episodes_completed = 15
        self.stage.successful_episodes = 10
        self.stage.min_episodes = 10
        self.stage.target_success_rate = 0.7
        self.stage.get_success_rate.return_value = 0.67  # Just below target
        self.stage.is_skill_mastered.side_effect = lambda skill, level=MasteryLevel.INTERMEDIATE: (
            skill == SkillType.STATIC_BALANCE
        )
        
        # Create test episode history
        self.recent_episodes = []
        for i in range(10):
            episode = Mock()
            episode.is_successful.return_value = i >= 3  # 7/10 successful
            episode.performance_metrics = PerformanceMetrics(
                success_rate=0.8 if i >= 3 else 0.2,
                average_reward=10.0 if i >= 3 else 3.0
            )
            self.recent_episodes.append(episode)
    
    def test_evaluate_advancement_readiness_ready(self):
        """Test advancement evaluation when robot is ready."""
        # Set up stage to meet all criteria
        self.stage.get_success_rate.return_value = 0.8  # Above target
        # Mock is_skill_mastered to accept skill parameter and return True for all skills
        self.stage.is_skill_mastered.side_effect = lambda skill, level=MasteryLevel.INTERMEDIATE: True
        
        decision = self.service.evaluate_advancement_readiness(
            self.robot, self.stage, self.recent_episodes
        )
        
        assert isinstance(decision, AdvancementDecision)
        assert decision.should_advance
        assert decision.confidence_score > 0.7
        assert len(decision.success_criteria_met) > 0
        assert len(decision.remaining_requirements) == 0
    
    def test_evaluate_advancement_readiness_not_ready(self):
        """Test advancement evaluation when robot is not ready."""
        # Set up stage with unmet criteria
        self.stage.get_success_rate.return_value = 0.5  # Below target
        self.stage.is_skill_mastered.side_effect = lambda skill, level=MasteryLevel.INTERMEDIATE: False
        
        decision = self.service.evaluate_advancement_readiness(
            self.robot, self.stage, self.recent_episodes
        )
        
        assert isinstance(decision, AdvancementDecision)
        assert not decision.should_advance
        assert decision.confidence_score >= 0.0
        assert len(decision.remaining_requirements) > 0
    
    def test_evaluate_advancement_confidence_scoring(self):
        """Test confidence scoring in advancement decisions."""
        # Test high confidence scenario
        self.stage.episodes_completed = 50  # Many episodes
        self.stage.get_success_rate.return_value = 0.9  # High success rate
        self.stage.is_skill_mastered.return_value = True
        
        high_confidence_decision = self.service.evaluate_advancement_readiness(
            self.robot, self.stage, self.recent_episodes
        )
        
        # Test low confidence scenario
        self.stage.episodes_completed = 8  # Few episodes
        self.stage.get_success_rate.return_value = 0.71  # Just above target
        
        low_confidence_decision = self.service.evaluate_advancement_readiness(
            self.robot, self.stage, self.recent_episodes
        )
        
        assert high_confidence_decision.confidence_score > low_confidence_decision.confidence_score
    
    def test_recommend_difficulty_adjustment_increase(self):
        """Test difficulty adjustment recommendation for high performers."""
        # Create high-performing episode history
        high_performance_episodes = []
        for i in range(10):
            episode = Mock()
            episode.is_successful.return_value = True  # 100% success
            episode.performance_metrics = PerformanceMetrics(
                success_rate=0.95,
                average_reward=15.0,
                learning_progress=0.8
            )
            high_performance_episodes.append(episode)
        
        adjustment = self.service.recommend_difficulty_adjustment(
            self.stage, high_performance_episodes
        )
        
        assert isinstance(adjustment, DifficultyAdjustment)
        assert adjustment.adjustment_type == 'increase'
        assert adjustment.magnitude > 0.0
        assert 'success_rate_too_high' in [r.reason for r in adjustment.recommendations]
    
    def test_recommend_difficulty_adjustment_decrease(self):
        """Test difficulty adjustment recommendation for low performers."""
        # Create low-performing episode history
        low_performance_episodes = []
        for i in range(10):
            episode = Mock()
            episode.is_successful.return_value = False  # 0% success
            episode.performance_metrics = PerformanceMetrics(
                success_rate=0.1,
                average_reward=2.0,
                learning_progress=-0.1
            )
            low_performance_episodes.append(episode)
        
        adjustment = self.service.recommend_difficulty_adjustment(
            self.stage, low_performance_episodes
        )
        
        assert isinstance(adjustment, DifficultyAdjustment)
        assert adjustment.adjustment_type == 'decrease'
        assert adjustment.magnitude > 0.0
        assert 'success_rate_too_low' in [r.reason for r in adjustment.recommendations]
    
    def test_recommend_difficulty_adjustment_maintain(self):
        """Test difficulty adjustment when performance is optimal."""
        # Create optimal performance episode history
        optimal_episodes = []
        for i in range(10):
            episode = Mock()
            episode.is_successful.return_value = i >= 3  # 70% success (at target)
            episode.performance_metrics = PerformanceMetrics(
                success_rate=0.7,
                average_reward=8.0,
                learning_progress=0.1
            )
            optimal_episodes.append(episode)
        
        adjustment = self.service.recommend_difficulty_adjustment(
            self.stage, optimal_episodes
        )
        
        assert isinstance(adjustment, DifficultyAdjustment)
        assert adjustment.adjustment_type == 'maintain'
        assert adjustment.magnitude == 0.0
    
    def test_identify_skill_gaps(self):
        """Test skill gap identification."""
        # Create robot with partial skill mastery
        self.robot.learned_skills = {
            SkillType.STATIC_BALANCE: LocomotionSkill(
                skill_type=SkillType.STATIC_BALANCE,
                mastery_level=MasteryLevel.EXPERT,
                proficiency_score=0.9
            )
            # Missing FORWARD_WALKING skill
        }
        
        gaps = self.service.identify_skill_gaps(self.robot, self.stage)
        
        assert isinstance(gaps, list)
        assert len(gaps) > 0
        
        # Should identify missing forward walking skill
        walking_gaps = [g for g in gaps if g['skill'] == SkillType.FORWARD_WALKING]
        assert len(walking_gaps) > 0
        assert walking_gaps[0]['gap_type'] == 'missing'
    
    def test_predict_learning_trajectory(self):
        """Test learning trajectory prediction."""
        # Create progression data
        performance_history = [
            PerformanceMetrics(success_rate=0.2, average_reward=3.0),
            PerformanceMetrics(success_rate=0.4, average_reward=5.0),
            PerformanceMetrics(success_rate=0.6, average_reward=7.0),
            PerformanceMetrics(success_rate=0.7, average_reward=9.0),
        ]
        
        prediction = self.service.predict_learning_trajectory(
            self.robot, self.stage, performance_history
        )
        
        assert isinstance(prediction, dict)
        assert 'estimated_episodes_to_mastery' in prediction
        assert 'predicted_success_rate' in prediction
        assert 'confidence_interval' in prediction
        assert 'trajectory_trend' in prediction
        
        assert prediction['estimated_episodes_to_mastery'] > 0
        assert 0.0 <= prediction['predicted_success_rate'] <= 1.0
        assert prediction['trajectory_trend'] in ['improving', 'stable', 'declining']
    
    def test_calculate_curriculum_efficiency(self):
        """Test curriculum efficiency calculation."""
        # Create mock curriculum plan
        curriculum_plan = Mock()
        
        # Create stages with different completion rates
        stage1 = Mock()
        stage1.expected_duration_episodes = 20
        stage1.episodes_completed = 18  # Efficient
        stage1.get_success_rate.return_value = 0.8
        
        stage2 = Mock()
        stage2.expected_duration_episodes = 30
        stage2.episodes_completed = 45  # Inefficient
        stage2.get_success_rate.return_value = 0.6
        
        curriculum_plan.stages = [stage1, stage2]
        
        efficiency = self.service.calculate_curriculum_efficiency(curriculum_plan)
        
        assert isinstance(efficiency, dict)
        assert 'overall_efficiency' in efficiency
        assert 'stage_efficiencies' in efficiency
        assert 'time_to_completion_ratio' in efficiency
        
        assert 0.0 <= efficiency['overall_efficiency'] <= 1.0
        assert len(efficiency['stage_efficiencies']) == 2
    
    def test_optimize_stage_sequence(self):
        """Test stage sequence optimization."""
        # Create stages with dependencies
        balance_stage = Mock()
        balance_stage.target_skills = {SkillType.STATIC_BALANCE}
        balance_stage.prerequisite_skills = set()
        balance_stage.stage_id = "balance"
        
        walking_stage = Mock()
        walking_stage.target_skills = {SkillType.FORWARD_WALKING}
        walking_stage.prerequisite_skills = {SkillType.STATIC_BALANCE}
        walking_stage.stage_id = "walking"
        
        turning_stage = Mock()
        turning_stage.target_skills = {SkillType.TURNING}
        turning_stage.prerequisite_skills = {SkillType.FORWARD_WALKING}
        turning_stage.stage_id = "turning"
        
        stages = [turning_stage, walking_stage, balance_stage]  # Wrong order
        
        optimized_sequence = self.service.optimize_stage_sequence(stages, self.robot)
        
        assert len(optimized_sequence) == 3
        
        # Should be reordered according to dependencies
        stage_ids = [stage.stage_id for stage in optimized_sequence]
        balance_index = stage_ids.index("balance")
        walking_index = stage_ids.index("walking")
        turning_index = stage_ids.index("turning")
        
        assert balance_index < walking_index  # Balance before walking
        assert walking_index < turning_index  # Walking before turning