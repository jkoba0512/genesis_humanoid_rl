"""
Curriculum progression domain service.
Encapsulates business logic for curriculum advancement and difficulty adaptation.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

from ..model.value_objects import PerformanceMetrics, SkillType, MasteryLevel
from ..model.entities import CurriculumStage, AdvancementCriteria
from ..model.aggregates import LearningSession, HumanoidRobot, CurriculumPlan

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AdvancementDecision:
    """Decision on curriculum advancement."""

    should_advance: bool
    confidence_score: float  # 0.0 to 1.0
    success_criteria_met: List[str]
    remaining_requirements: List[str]


@dataclass(frozen=True)
class DifficultyAdjustmentRecommendation:
    """Individual difficulty adjustment recommendation."""

    reason: str
    magnitude: float
    parameter: str


@dataclass(frozen=True)
class DifficultyAdjustment:
    """Difficulty adjustment recommendation."""

    adjustment_type: str  # 'increase', 'decrease', 'maintain'
    magnitude: float  # 0.0 to 1.0
    recommendations: List[DifficultyAdjustmentRecommendation]


@dataclass(frozen=True)
class AdvancementResult:
    """Result of curriculum advancement evaluation."""

    decision: AdvancementDecision
    confidence: float  # 0.0 to 1.0
    reasoning: List[str]
    recommendations: List[str]
    next_stage_prediction: Optional[int] = None


class CurriculumProgressionService:
    """
    Domain service for curriculum advancement business logic.

    Encapsulates complex rules for:
    - Stage advancement decisions
    - Difficulty adaptation
    - Performance threshold management
    - Learning progress assessment
    """

    def evaluate_advancement_readiness(
        self, robot: HumanoidRobot, stage: CurriculumStage, recent_episodes: List[Any]
    ) -> AdvancementDecision:
        """Evaluate if robot is ready for curriculum advancement."""
        success_criteria_met = []
        remaining_requirements = []

        # Check episode count
        if stage.episodes_completed >= stage.min_episodes:
            success_criteria_met.append("episode_count")
        else:
            remaining_requirements.append("episode_count")

        # Check success rate
        if stage.get_success_rate() >= stage.target_success_rate:
            success_criteria_met.append("success_rate")
        else:
            remaining_requirements.append("success_rate")

        # Check skill mastery
        try:
            target_skills = getattr(stage, "target_skills", set())
            if not target_skills:
                # No target skills means mastery requirement is automatically met
                all_skills_mastered = True
            else:
                all_skills_mastered = all(
                    stage.is_skill_mastered(skill) for skill in target_skills
                )
        except (AttributeError, TypeError) as e:
            logger.warning(f"Error checking skill mastery: {e}")
            all_skills_mastered = False

        if all_skills_mastered:
            success_criteria_met.append("skill_mastery")
        else:
            remaining_requirements.append("skill_mastery")

        # Calculate confidence based on criteria met
        confidence_score = len(success_criteria_met) / (
            len(success_criteria_met) + len(remaining_requirements)
        )
        confidence_score = max(
            confidence_score, len(success_criteria_met) / 3.0
        )  # Base confidence

        # Adjust confidence based on episode count
        if stage.episodes_completed > stage.min_episodes * 2:
            confidence_score = min(confidence_score + 0.2, 1.0)

        should_advance = len(remaining_requirements) == 0

        return AdvancementDecision(
            should_advance=should_advance,
            confidence_score=confidence_score,
            success_criteria_met=success_criteria_met,
            remaining_requirements=remaining_requirements,
        )

    def recommend_difficulty_adjustment(
        self, stage: CurriculumStage, recent_episodes: List[Any]
    ) -> DifficultyAdjustment:
        """Recommend difficulty adjustments based on performance."""
        if not recent_episodes:
            return DifficultyAdjustment(
                adjustment_type="maintain", magnitude=0.0, recommendations=[]
            )

        # Calculate recent success rate
        successful_count = sum(
            1 for episode in recent_episodes if episode.is_successful()
        )
        success_rate = successful_count / len(recent_episodes)

        # Calculate average learning progress
        avg_learning_progress = sum(
            episode.performance_metrics.learning_progress
            for episode in recent_episodes
            if hasattr(episode.performance_metrics, "learning_progress")
            and episode.performance_metrics.learning_progress is not None
        ) / max(1, len(recent_episodes))

        recommendations = []

        # High performance - increase difficulty
        if success_rate > 0.9 and avg_learning_progress > 0.5:
            recommendations.append(
                DifficultyAdjustmentRecommendation(
                    reason="success_rate_too_high",
                    magnitude=0.2,
                    parameter="target_success_rate",
                )
            )
            return DifficultyAdjustment(
                adjustment_type="increase",
                magnitude=0.2,
                recommendations=recommendations,
            )

        # Low performance - decrease difficulty
        elif success_rate < 0.3 or avg_learning_progress < -0.1:
            recommendations.append(
                DifficultyAdjustmentRecommendation(
                    reason="success_rate_too_low",
                    magnitude=0.3,
                    parameter="target_success_rate",
                )
            )
            return DifficultyAdjustment(
                adjustment_type="decrease",
                magnitude=0.3,
                recommendations=recommendations,
            )

        # Maintain current difficulty
        else:
            return DifficultyAdjustment(
                adjustment_type="maintain", magnitude=0.0, recommendations=[]
            )

    def identify_skill_gaps(
        self, robot: HumanoidRobot, stage: CurriculumStage
    ) -> List[Dict[str, Any]]:
        """Identify gaps in robot's skill mastery for the stage."""
        gaps = []

        for skill_type in stage.target_skills:
            if skill_type not in robot.learned_skills:
                gaps.append(
                    {
                        "skill": skill_type,
                        "gap_type": "missing",
                        "current_proficiency": 0.0,
                        "target_proficiency": 0.7,
                        "priority": "high",
                    }
                )
            else:
                current_skill = robot.learned_skills[skill_type]
                if current_skill.proficiency_score < 0.7:
                    gaps.append(
                        {
                            "skill": skill_type,
                            "gap_type": "insufficient_proficiency",
                            "current_proficiency": current_skill.proficiency_score,
                            "target_proficiency": 0.7,
                            "priority": "medium",
                        }
                    )

        return gaps

    def predict_learning_trajectory(
        self,
        robot: HumanoidRobot,
        stage: CurriculumStage,
        performance_history: List[PerformanceMetrics],
    ) -> Dict[str, Any]:
        """Predict learning trajectory based on historical performance."""
        if len(performance_history) < 2:
            return {
                "estimated_episodes_to_mastery": 50,
                "predicted_success_rate": 0.5,
                "confidence_interval": (0.3, 0.7),
                "trajectory_trend": "unknown",
            }

        # Calculate trend
        recent_performance = [p.success_rate for p in performance_history[-3:]]
        older_performance = (
            [p.success_rate for p in performance_history[:-3]]
            if len(performance_history) > 3
            else recent_performance
        )

        recent_avg = sum(recent_performance) / len(recent_performance)
        older_avg = sum(older_performance) / len(older_performance)

        trend_direction = (
            "improving"
            if recent_avg > older_avg
            else "declining" if recent_avg < older_avg else "stable"
        )

        # Estimate episodes to mastery
        current_success_rate = recent_avg
        target_success_rate = stage.target_success_rate

        if current_success_rate >= target_success_rate:
            episodes_to_mastery = 5  # Almost there
        else:
            improvement_rate = max(
                (recent_avg - older_avg) / len(performance_history), 0.01
            )
            episodes_needed = (
                target_success_rate - current_success_rate
            ) / improvement_rate
            episodes_to_mastery = max(10, min(int(episodes_needed), 200))

        return {
            "estimated_episodes_to_mastery": episodes_to_mastery,
            "predicted_success_rate": min(recent_avg + 0.1, 1.0),
            "confidence_interval": (
                max(recent_avg - 0.2, 0.0),
                min(recent_avg + 0.2, 1.0),
            ),
            "trajectory_trend": trend_direction,
        }

    def calculate_curriculum_efficiency(
        self, curriculum_plan: CurriculumPlan
    ) -> Dict[str, Any]:
        """Calculate efficiency metrics for curriculum plan."""
        if not curriculum_plan.stages:
            return {
                "overall_efficiency": 0.0,
                "stage_efficiencies": [],
                "time_to_completion_ratio": 1.0,
            }

        stage_efficiencies = []
        total_expected = 0
        total_actual = 0

        for stage in curriculum_plan.stages:
            expected_episodes = stage.expected_duration_episodes
            actual_episodes = stage.episodes_completed

            if expected_episodes > 0:
                efficiency = min(
                    expected_episodes / max(actual_episodes, 1), 2.0
                )  # Cap at 2x efficiency
            else:
                efficiency = 1.0

            stage_efficiencies.append(
                {
                    "stage_id": stage.stage_id,
                    "efficiency": efficiency,
                    "success_rate": stage.get_success_rate(),
                }
            )

            total_expected += expected_episodes
            total_actual += actual_episodes

        overall_efficiency = (
            total_expected / max(total_actual, 1) if total_actual > 0 else 1.0
        )

        return {
            "overall_efficiency": min(overall_efficiency, 2.0),
            "stage_efficiencies": stage_efficiencies,
            "time_to_completion_ratio": total_actual / max(total_expected, 1),
        }

    def optimize_stage_sequence(
        self, stages: List[CurriculumStage], robot: HumanoidRobot
    ) -> List[CurriculumStage]:
        """Optimize the sequence of curriculum stages based on dependencies."""
        # Simple topological sort based on prerequisites
        optimized = []
        remaining = stages.copy()

        while remaining:
            # Find stages with no unmet prerequisites
            ready_stages = []
            for stage in remaining:
                prerequisites_met = True
                for prereq_skill in stage.prerequisite_skills:
                    # Check if prerequisite skill is covered by already-included stages
                    if not any(prereq_skill in s.target_skills for s in optimized):
                        prerequisites_met = False
                        break

                if prerequisites_met:
                    ready_stages.append(stage)

            if not ready_stages:
                # Break circular dependencies - add remaining stages
                ready_stages = remaining

            # Sort ready stages by difficulty/order
            ready_stages.sort(key=lambda s: s.order)

            # Add the first ready stage
            next_stage = ready_stages[0]
            optimized.append(next_stage)
            remaining.remove(next_stage)

        return optimized

    def should_advance_stage(
        self,
        session: LearningSession,
        current_stage: CurriculumStage,
        robot: HumanoidRobot,
    ) -> AdvancementResult:
        """
        Determine if session should advance to next curriculum stage.

        Core business rule: Multiple criteria must be met for advancement.
        """
        reasoning = []
        recommendations = []

        # Check basic advancement criteria
        basic_criteria_met = current_stage.can_advance()
        if not basic_criteria_met:
            remaining = current_stage.get_remaining_requirements()
            reasoning.append(f"Basic criteria not met: {remaining}")

            decision = AdvancementDecision.CONTINUE
            if remaining.get("success_rate_gap", 0) > 0.3:
                decision = AdvancementDecision.ADJUST_DIFFICULTY
                recommendations.append(
                    "Consider reducing difficulty to improve success rate"
                )

            return AdvancementResult(
                decision=decision,
                confidence=0.8,
                reasoning=reasoning,
                recommendations=recommendations,
            )

        # Check robot readiness
        robot_readiness = self._assess_robot_readiness(robot, current_stage)
        reasoning.append(f"Robot readiness score: {robot_readiness:.2f}")

        # Check learning trajectory
        learning_momentum = self._calculate_learning_momentum(session)
        reasoning.append(f"Learning momentum: {learning_momentum:.2f}")

        # Make advancement decision
        if robot_readiness >= 0.7 and learning_momentum >= 0.5:
            decision = AdvancementDecision.ADVANCE
            confidence = min((robot_readiness + learning_momentum) / 2.0, 0.95)
            reasoning.append("All criteria met for advancement")
        elif robot_readiness >= 0.5:
            decision = AdvancementDecision.CONTINUE
            confidence = 0.7
            reasoning.append("Criteria partially met, continue current stage")
            recommendations.append("Focus on skill mastery before advancement")
        else:
            decision = AdvancementDecision.REPEAT
            confidence = 0.8
            reasoning.append("Robot not ready for advancement")
            recommendations.append("Repeat current stage with adjusted parameters")

        return AdvancementResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            recommendations=recommendations,
            next_stage_prediction=(
                session.current_stage_index + 1
                if decision == AdvancementDecision.ADVANCE
                else None
            ),
        )

    def calculate_readiness_score(
        self, robot: HumanoidRobot, target_stage: CurriculumStage
    ) -> float:
        """
        Calculate robot's readiness for a specific curriculum stage.

        Business rule: Readiness based on skill mastery and performance history.
        """
        if not target_stage.prerequisite_skills:
            return 1.0  # No prerequisites

        # Check prerequisite skill mastery
        prerequisite_score = 0.0
        for skill in target_stage.prerequisite_skills:
            skill_proficiency = robot.get_skill_proficiency(skill)
            prerequisite_score += skill_proficiency

        prerequisite_score /= len(target_stage.prerequisite_skills)

        # Check recent performance trends
        performance_trend = self._calculate_performance_trend(robot)

        # Combine scores
        readiness_score = prerequisite_score * 0.7 + performance_trend * 0.3

        return min(readiness_score, 1.0)

    def adapt_difficulty(
        self,
        current_stage: CurriculumStage,
        performance_history: List[PerformanceMetrics],
    ) -> Dict[str, Any]:
        """
        Adapt curriculum difficulty based on performance.

        Business rule: Maintain optimal challenge level for learning.
        """
        if not performance_history:
            return {"difficulty_adjustment": "none", "reason": "insufficient_data"}

        # Analyze recent performance
        recent_performance = performance_history[-10:]  # Last 10 records
        avg_performance = sum(
            p.get_overall_performance() for p in recent_performance
        ) / len(recent_performance)

        adaptations = {}

        # Performance-based adjustments
        if avg_performance > 0.85:
            # Too easy, increase difficulty
            adaptations.update(
                {
                    "difficulty_adjustment": "increase",
                    "target_success_rate": min(
                        current_stage.target_success_rate + 0.1, 0.9
                    ),
                    "min_episodes": max(current_stage.min_episodes - 5, 10),
                    "reason": "performance_too_high",
                }
            )
        elif avg_performance < 0.3:
            # Too hard, decrease difficulty
            adaptations.update(
                {
                    "difficulty_adjustment": "decrease",
                    "target_success_rate": max(
                        current_stage.target_success_rate - 0.15, 0.5
                    ),
                    "min_episodes": current_stage.min_episodes + 10,
                    "reason": "performance_too_low",
                }
            )
        else:
            # Appropriate difficulty
            adaptations.update(
                {
                    "difficulty_adjustment": "maintain",
                    "reason": "optimal_challenge_level",
                }
            )

        # Learning rate adjustments
        learning_rates = [p.learning_progress for p in recent_performance]
        avg_learning_rate = sum(learning_rates) / len(learning_rates)

        if avg_learning_rate < 0.1:
            adaptations["suggested_action"] = "increase_training_variation"
        elif avg_learning_rate > 0.8:
            adaptations["suggested_action"] = "accelerate_progression"

        return adaptations

    def predict_advancement_timeline(
        self, session: LearningSession, curriculum: CurriculumPlan
    ) -> Dict[str, Any]:
        """
        Predict timeline for curriculum completion.

        Business logic for learning progress estimation.
        """
        if session.current_stage_index >= len(curriculum.stages):
            return {"status": "completed", "remaining_time": 0}

        # Calculate current learning rate
        learning_rate = session.get_learning_progress()

        # Estimate time per remaining stage
        remaining_stages = len(curriculum.stages) - session.current_stage_index

        if learning_rate <= 0:
            return {"status": "stalled", "remaining_time": None}

        # Estimate based on current stage progress
        current_stage = curriculum.stages[session.current_stage_index]
        current_progress = current_stage.get_progress_percentage()

        # Simple linear projection (could be more sophisticated)
        episodes_per_stage = 50  # Average estimate
        estimated_episodes_remaining = (
            remaining_stages * episodes_per_stage * (1.0 - current_progress / 100.0)
        )

        # Adjust for learning rate
        adjusted_episodes = estimated_episodes_remaining / max(learning_rate, 0.1)

        return {
            "status": "in_progress",
            "remaining_stages": remaining_stages,
            "estimated_episodes": int(adjusted_episodes),
            "confidence": min(learning_rate, 0.8),
            "current_stage_progress": current_progress,
        }

    # Private helper methods

    def _assess_robot_readiness(
        self, robot: HumanoidRobot, stage: CurriculumStage
    ) -> float:
        """Assess robot's readiness for stage requirements."""
        if not stage.target_skills:
            return 1.0

        # Check skill preparation
        readiness_scores = []
        for skill in stage.target_skills:
            current_proficiency = robot.get_skill_proficiency(skill)

            # Higher readiness if already have some proficiency
            if current_proficiency > 0.3:
                readiness_scores.append(0.8)
            elif robot.can_learn_skill(skill):
                readiness_scores.append(0.6)
            else:
                readiness_scores.append(0.2)

        return sum(readiness_scores) / len(readiness_scores)

    def _calculate_learning_momentum(self, session: LearningSession) -> float:
        """Calculate learning momentum from recent episodes."""
        if len(session.episodes) < 10:
            return 0.5  # Neutral momentum for insufficient data

        recent_episodes = session.episodes[-10:]
        early_episodes = (
            session.episodes[-20:-10]
            if len(session.episodes) >= 20
            else session.episodes[:-10]
        )

        if not early_episodes:
            return 0.5

        # Compare recent vs early performance
        recent_success_rate = sum(
            1 for ep in recent_episodes if ep.is_successful()
        ) / len(recent_episodes)
        early_success_rate = sum(
            1 for ep in early_episodes if ep.is_successful()
        ) / len(early_episodes)

        # Momentum based on improvement
        improvement = recent_success_rate - early_success_rate
        momentum = 0.5 + improvement  # Center around 0.5

        return max(0.0, min(1.0, momentum))

    def _calculate_performance_trend(self, robot: HumanoidRobot) -> float:
        """Calculate performance trend from robot's history."""
        if len(robot.performance_history) < 5:
            return 0.5  # Neutral trend

        recent_performance = robot.performance_history[-5:]
        performance_scores = [p.get_overall_performance() for p in recent_performance]

        # Simple linear trend
        if len(performance_scores) < 2:
            return 0.5

        # Calculate slope of performance
        x = list(range(len(performance_scores)))
        y = performance_scores

        n = len(x)
        slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (
            n * sum(xi**2 for xi in x) - sum(x) ** 2
        )

        # Convert slope to 0-1 scale
        trend_score = 0.5 + slope * 2.0  # Amplify slope effect
        return max(0.0, min(1.0, trend_score))
