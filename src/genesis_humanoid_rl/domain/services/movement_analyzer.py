"""
Movement quality analysis domain service.
Provides business logic for analyzing robot movement patterns and gait quality.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from enum import Enum

from ..model.value_objects import (
    GaitPattern,
    MovementTrajectory,
    MotionCommand,
    MotionType,
    LocomotionSkill,
    SkillType,
    MasteryLevel,
)
from ...protocols import RobotState

logger = logging.getLogger(__name__)


class MovementQuality(Enum):
    """Overall movement quality assessment."""

    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


@dataclass(frozen=True)
class GaitAnalysisResult:
    """Result of gait pattern analysis."""

    gait_pattern: GaitPattern
    quality_score: float  # 0.0 to 1.0
    quality_rating: MovementQuality
    stability_assessment: str
    efficiency_assessment: str
    recommendations: List[str]

    @property
    def overall_score(self) -> float:
        """Alias for quality_score to maintain compatibility."""
        return self.quality_score


@dataclass(frozen=True)
class MovementPatternAnalysis:
    """Analysis of overall movement patterns."""

    pattern_type: str
    consistency_score: float  # 0.0 to 1.0
    learning_indicators: List[str]
    improvement_areas: List[str]
    strengths: List[str]


@dataclass(frozen=True)
class StabilityAnalysisResult:
    """Result of movement stability analysis."""

    overall_score: float  # 0.0 to 1.0
    velocity_consistency: float  # 0.0 to 1.0
    acceleration_smoothness: float  # 0.0 to 1.0
    balance_stability: float  # 0.0 to 1.0
    analysis_notes: str

    def is_stable(self, threshold: float = 0.7) -> bool:
        """Check if movement is considered stable."""
        return self.overall_score >= threshold


class MovementQualityAnalyzer:
    """
    Domain service for analyzing robot movement quality and patterns.

    Encapsulates complex business logic for:
    - Gait stability analysis
    - Energy efficiency assessment
    - Movement pattern recognition
    - Skill progression evaluation
    """

    def __init__(self):
        self._movement_history: List[MovementTrajectory] = []
        self._gait_history: List[GaitPattern] = []

    def analyze_gait_stability(
        self, trajectory: MovementTrajectory
    ) -> StabilityAnalysisResult:
        """
        Analyze gait stability from movement trajectory.

        Core business logic for determining walking quality.
        """
        # Handle insufficient data case
        if (
            len(trajectory.positions) < 3
            or (trajectory.timestamps[-1] - trajectory.timestamps[0]) <= 0
        ):
            return StabilityAnalysisResult(
                overall_score=0.0,
                velocity_consistency=0.0,
                acceleration_smoothness=0.0,
                balance_stability=0.0,
                analysis_notes="Insufficient data for analysis",
            )

        # Extract gait characteristics from trajectory
        gait_pattern = self._extract_gait_pattern(trajectory)

        # Analyze stability components
        stability_score = self._calculate_stability_score(trajectory, gait_pattern)
        efficiency_score = gait_pattern.energy_efficiency
        symmetry_score = gait_pattern.symmetry_score

        # Calculate overall quality
        quality_score = (
            stability_score * 0.4 + efficiency_score * 0.3 + symmetry_score * 0.3
        )

        # Determine quality rating
        quality_rating = self._determine_quality_rating(quality_score)

        # Generate assessments and recommendations
        stability_assessment = self._assess_stability(stability_score, gait_pattern)
        efficiency_assessment = self._assess_efficiency(efficiency_score, gait_pattern)
        recommendations = self._generate_gait_recommendations(
            gait_pattern, stability_score, efficiency_score, symmetry_score
        )

        # Store for historical analysis
        self._gait_history.append(gait_pattern)
        if len(self._gait_history) > 100:
            self._gait_history = self._gait_history[-100:]

        # Store gait analysis results as well
        gait_analysis = GaitAnalysisResult(
            gait_pattern=gait_pattern,
            quality_score=quality_score,
            quality_rating=quality_rating,
            stability_assessment=stability_assessment,
            efficiency_assessment=efficiency_assessment,
            recommendations=recommendations,
        )

        # Calculate velocity consistency more accurately from trajectory
        velocity_consistency = self._calculate_velocity_consistency(trajectory)
        acceleration_smoothness = self._calculate_acceleration_smoothness(trajectory)

        # Return StabilityAnalysisResult as expected by tests
        return StabilityAnalysisResult(
            overall_score=quality_score,
            velocity_consistency=velocity_consistency,
            acceleration_smoothness=acceleration_smoothness,
            balance_stability=(
                gait_pattern.stability_margin / max(0.1, gait_pattern.stability_margin)
                if gait_pattern.stability_margin > 0
                else 0.0
            ),
            analysis_notes=stability_assessment,
        )

    def assess_energy_efficiency(
        self,
        joint_commands: List[np.ndarray],
        distance_covered: float,
        time_elapsed: float,
    ) -> Tuple[float, str]:
        """
        Assess energy efficiency of robot movement.

        Business rule: Efficiency = useful work / energy expended
        """
        if time_elapsed <= 0 or distance_covered <= 0:
            return 0.0, "No meaningful movement detected"

        # Calculate energy expenditure (simplified model)
        total_energy = 0.0
        for commands in joint_commands:
            # Energy approximated by sum of squared joint commands
            energy_per_step = np.sum(np.square(commands))
            total_energy += energy_per_step

        # Normalize by distance and time
        if total_energy == 0:
            efficiency_score = 0.0
            assessment = "No energy expenditure detected"
        else:
            # Higher distance per energy is better
            raw_efficiency = distance_covered / total_energy

            # Normalize to 0-1 scale (based on empirical thresholds)
            max_expected_efficiency = 0.1  # Typical good efficiency
            efficiency_score = min(raw_efficiency / max_expected_efficiency, 1.0)

            assessment = self._generate_efficiency_assessment(
                efficiency_score, distance_covered, time_elapsed
            )

        return efficiency_score, assessment

    def detect_movement_patterns(
        self, state_history: List[RobotState]
    ) -> MovementPatternAnalysis:
        """
        Detect and analyze movement patterns from robot state history.

        Business logic for identifying learned behaviors and skill development.
        """
        if len(state_history) < 10:
            return MovementPatternAnalysis(
                pattern_type="insufficient_data",
                consistency_score=0.0,
                learning_indicators=[],
                improvement_areas=["Need more movement data"],
                strengths=[],
            )

        # Analyze movement consistency
        consistency_score = self._analyze_movement_consistency(state_history)

        # Identify pattern type
        pattern_type = self._identify_primary_pattern(state_history)

        # Detect learning indicators
        learning_indicators = self._detect_learning_indicators(state_history)

        # Identify improvement areas
        improvement_areas = self._identify_improvement_areas(state_history)

        # Identify strengths
        strengths = self._identify_movement_strengths(state_history)

        return MovementPatternAnalysis(
            pattern_type=pattern_type,
            consistency_score=consistency_score,
            learning_indicators=learning_indicators,
            improvement_areas=improvement_areas,
            strengths=strengths,
        )

    def evaluate_skill_progression(
        self, skill_type: SkillType, recent_trajectories: List[MovementTrajectory]
    ) -> Dict[str, Any]:
        """
        Evaluate progression in a specific locomotion skill.

        Business logic for tracking skill development over time.
        """
        if not recent_trajectories:
            return {
                "skill_type": skill_type.value,
                "progression_score": 0.0,
                "trend": "no_data",
                "mastery_indicators": [],
                "next_milestones": [],
            }

        # Analyze skill-specific metrics
        skill_metrics = self._calculate_skill_metrics(skill_type, recent_trajectories)

        # Calculate progression trend
        progression_trend = self._calculate_progression_trend(skill_metrics)

        # Identify mastery indicators
        mastery_indicators = self._identify_mastery_indicators(
            skill_type, skill_metrics
        )

        # Determine next milestones
        next_milestones = self._determine_next_milestones(skill_type, skill_metrics)

        return {
            "skill_type": skill_type.value,
            "progression_score": skill_metrics.get("overall_score", 0.0),
            "trend": progression_trend,
            "mastery_indicators": mastery_indicators,
            "next_milestones": next_milestones,
            "detailed_metrics": skill_metrics,
        }

    def extract_gait_pattern_from_trajectory(
        self, trajectory: MovementTrajectory
    ) -> GaitPattern:
        """
        Public method to extract gait pattern from trajectory.

        Refactored to expose private method for testing and external use.
        """
        return self._extract_gait_pattern(trajectory)

    def assess_movement_quality(self, trajectory_or_episode) -> Dict[str, Any]:
        """
        Comprehensive movement quality assessment.

        Provides overall movement analysis combining multiple factors.
        Can take either a trajectory or an episode with trajectory.
        """
        # Handle episode with trajectory vs direct trajectory
        if hasattr(trajectory_or_episode, "movement_trajectory"):
            trajectory = trajectory_or_episode.movement_trajectory
        else:
            trajectory = trajectory_or_episode

        if (
            not trajectory
            or not hasattr(trajectory, "positions")
            or len(trajectory.positions) < 2
        ):
            return {
                "overall_quality_score": 0.0,
                "gait_analysis": None,
                "stability_analysis": None,
                "recommendations": ["Insufficient trajectory data"],
            }

        # Perform gait analysis
        gait_analysis = self.analyze_gait_stability(trajectory)

        # gait_analysis is already a StabilityAnalysisResult, use it directly
        stability_analysis = gait_analysis

        # Calculate efficiency metrics
        efficiency_score = self.calculate_energy_efficiency(trajectory)
        efficiency_metrics = {
            "energy_efficiency": efficiency_score,
            "movement_smoothness": trajectory.get_smoothness_score(),
            "velocity_consistency": gait_analysis.velocity_consistency,
        }

        return {
            "overall_quality_score": gait_analysis.overall_score,
            "gait_analysis": gait_analysis,
            "stability_analysis": stability_analysis,
            "efficiency_metrics": efficiency_metrics,
            "recommendations": [],  # StabilityAnalysisResult doesn't have recommendations
        }

    def identify_movement_anomalies(
        self, trajectory: MovementTrajectory
    ) -> List[Dict[str, Any]]:
        """
        Identify anomalies in a single movement trajectory.

        Analyzes trajectory to detect unusual or problematic movements.
        """
        anomalies = []

        if not trajectory or len(trajectory.positions) < 2:
            return anomalies

        # Check for excessive velocity changes
        if len(trajectory.positions) >= 3:
            velocities = []
            for j in range(1, len(trajectory.positions)):
                pos_prev = np.array(trajectory.positions[j - 1])
                pos_curr = np.array(trajectory.positions[j])
                dt = 0.1  # Assume 10Hz
                velocity = np.linalg.norm(pos_curr - pos_prev) / dt
                velocities.append(velocity)

            if velocities:
                max_velocity = max(velocities)
                avg_velocity = np.mean(velocities)

                # Check for velocity spikes (more sensitive)
                if (
                    max_velocity > avg_velocity * 2.0 or max_velocity > 5.0
                ):  # Lower threshold
                    anomalies.append(
                        {
                            "type": "velocity_spike",
                            "severity": "high",
                            "description": f"Velocity spike detected: {max_velocity:.2f}m/s",
                        }
                    )

                # Also check for sudden acceleration between consecutive steps
                for j in range(1, len(velocities)):
                    velocity_change = abs(velocities[j] - velocities[j - 1])
                    if velocity_change > 3.0:  # Sudden acceleration change
                        anomalies.append(
                            {
                                "type": "velocity_spike",
                                "severity": "medium",
                                "description": f"Sudden velocity change: {velocity_change:.2f}m/s",
                            }
                        )
                        break  # Only report one

        # Check for height anomalies
        if len(trajectory.positions) >= 2:
            heights = [pos[2] for pos in trajectory.positions]
            min_height = min(heights)
            max_height = max(heights)
            height_range = max_height - min_height

            if min_height < 0.2:
                anomalies.append(
                    {
                        "type": "low_height",
                        "severity": "high",
                        "description": f"Robot too low: {min_height:.2f}m",
                    }
                )

            if max_height > 1.5:
                anomalies.append(
                    {
                        "type": "height_spike",
                        "severity": "medium",
                        "description": f"Robot too high: {max_height:.2f}m",
                    }
                )

            if height_range > 1.0:  # Large height variation
                anomalies.append(
                    {
                        "type": "height_spike",
                        "severity": "high",
                        "description": f"Large height variation: {height_range:.2f}m",
                    }
                )

        return anomalies

    def calculate_energy_efficiency(self, trajectory: MovementTrajectory) -> float:
        """
        Calculate energy efficiency score from trajectory.

        Refactored to take trajectory and calculate efficiency directly.
        """
        if not trajectory or len(trajectory.positions) < 2:
            return 0.0

        # Calculate distance and time
        distance_covered = trajectory.get_total_distance()
        time_elapsed = trajectory.timestamps[-1] - trajectory.timestamps[0]

        if time_elapsed <= 0 or distance_covered <= 0:
            return 0.0

        # Estimate energy from movement smoothness (simpler approach)
        smoothness = trajectory.get_smoothness_score()
        velocity = distance_covered / time_elapsed

        # Efficiency is based on smoothness and reasonable velocity
        # High smoothness = less energy waste
        # Moderate velocity = good efficiency
        velocity_efficiency = 1.0 - abs(velocity - 1.0) / 2.0  # Optimal around 1.0 m/s
        velocity_efficiency = max(0.0, velocity_efficiency)

        efficiency_score = smoothness * 0.7 + velocity_efficiency * 0.3
        return min(1.0, efficiency_score)

    def evaluate_balance_quality(self, state_history: List[Dict[str, Any]]) -> float:
        """
        Evaluate balance quality from robot state history.

        Returns a single float score representing balance quality.
        """
        if len(state_history) < 3:
            return 0.0

        # Extract positions from dict format
        positions = []
        orientations = []

        for state in state_history:
            if isinstance(state, dict):
                positions.append(state["position"])
                orientations.append(state["orientation"])
            else:
                # Handle RobotState objects
                positions.append(state.position)
                orientations.append(getattr(state, "orientation", [0, 0, 0, 1]))

        # Analyze height stability
        heights = [pos[2] for pos in positions]
        height_mean = np.mean(heights)
        height_var = np.var(heights)
        height_stability = max(
            0.0, 1.0 - height_var / 0.01
        )  # Normalize to expected variance

        # Analyze center of mass movement (simplified)
        positions_2d = [pos[:2] for pos in positions]  # X,Y only
        com_movements = []
        for i in range(1, len(positions_2d)):
            movement = np.linalg.norm(
                np.array(positions_2d[i]) - np.array(positions_2d[i - 1])
            )
            com_movements.append(movement)

        com_stability = 1.0
        if com_movements:
            avg_com_movement = np.mean(com_movements)
            com_stability = max(0.0, 1.0 - avg_com_movement / 0.1)  # Normalize to 10cm

        # Analyze orientation stability (quaternion deviation from upright)
        orientation_stability = 1.0
        if orientations:
            deviations = []
            for quat in orientations:
                # Calculate tilt from upright (quaternion [0,0,0,1])
                q = np.array(quat)
                # Simple tilt calculation using x and y components
                tilt = np.sqrt(q[0] ** 2 + q[1] ** 2)
                deviations.append(tilt)

            avg_deviation = np.mean(deviations)
            orientation_stability = max(
                0.0, 1.0 - avg_deviation / 0.5
            )  # Normalize to 0.5 rad tilt

        # Overall balance score
        balance_score = (
            height_stability * 0.4 + com_stability * 0.3 + orientation_stability * 0.3
        )

        return min(1.0, balance_score)

    def compare_gait_patterns(
        self, pattern1: GaitPattern, pattern2: GaitPattern
    ) -> float:
        """
        Compare two gait patterns and return similarity score.

        Returns a similarity score from 0.0 (completely different) to 1.0 (identical).
        """
        # Calculate differences in each dimension
        stride_length_diff = abs(pattern2.stride_length - pattern1.stride_length)
        frequency_diff = abs(pattern2.stride_frequency - pattern1.stride_frequency)
        stability_diff = abs(pattern2.stability_margin - pattern1.stability_margin)
        efficiency_diff = abs(pattern2.energy_efficiency - pattern1.energy_efficiency)
        symmetry_diff = abs(pattern2.symmetry_score - pattern1.symmetry_score)

        # Normalize differences to 0-1 scale (smaller max differences for more sensitivity)
        stride_similarity = max(0.0, 1.0 - stride_length_diff / 0.5)  # Max diff 0.5m
        frequency_similarity = max(0.0, 1.0 - frequency_diff / 1.5)  # Max diff 1.5 Hz
        stability_similarity = max(0.0, 1.0 - stability_diff / 0.1)  # Max diff 0.1m
        efficiency_similarity = max(0.0, 1.0 - efficiency_diff / 0.8)  # Max diff 0.8
        symmetry_similarity = max(0.0, 1.0 - symmetry_diff / 0.8)  # Max diff 0.8

        # Weighted average similarity
        overall_similarity = (
            stride_similarity * 0.25
            + frequency_similarity * 0.25
            + stability_similarity * 0.2
            + efficiency_similarity * 0.15
            + symmetry_similarity * 0.15
        )

        return min(1.0, overall_similarity)

    # Private helper methods

    def _extract_gait_pattern(self, trajectory: MovementTrajectory) -> GaitPattern:
        """Extract gait characteristics from movement trajectory."""
        # Simplified gait extraction (in practice, this would be more sophisticated)
        total_distance = trajectory.get_total_distance()
        total_time = trajectory.timestamps[-1] - trajectory.timestamps[0]

        if total_time <= 0:
            # Return a zero gait pattern for insufficient data
            zero_pattern = GaitPattern(
                stride_length=0.0,
                stride_frequency=0.0,
                step_height=0.0,
                stability_margin=0.0,
                energy_efficiency=0.0,
                symmetry_score=0.0,
            )
            return zero_pattern

        # Estimate gait parameters
        avg_velocity = total_distance / total_time
        estimated_stride_frequency = max(
            avg_velocity / 0.5, 0.1
        )  # Assume 0.5m stride length
        estimated_stride_length = (
            avg_velocity / estimated_stride_frequency
            if estimated_stride_frequency > 0
            else 0.0
        )

        # Calculate stability (based on trajectory smoothness)
        smoothness = trajectory.get_smoothness_score()
        stability_margin = smoothness * 0.1  # Convert to meters

        # Estimate efficiency (higher smoothness = better efficiency)
        energy_efficiency = smoothness

        # Estimate symmetry (placeholder - would need more sophisticated analysis)
        symmetry_score = min(smoothness + 0.2, 1.0)

        # Estimate step height (placeholder)
        step_height = 0.05  # 5cm default

        return GaitPattern(
            stride_length=estimated_stride_length,
            stride_frequency=estimated_stride_frequency,
            step_height=step_height,
            stability_margin=stability_margin,
            energy_efficiency=energy_efficiency,
            symmetry_score=symmetry_score,
        )

    def _calculate_stability_score(
        self, trajectory: MovementTrajectory, gait_pattern: GaitPattern
    ) -> float:
        """Calculate stability score from trajectory and gait data."""
        # Combine trajectory smoothness with gait stability margin
        trajectory_stability = trajectory.get_smoothness_score()
        gait_stability = min(
            gait_pattern.stability_margin / 0.1, 1.0
        )  # Normalize to 0.1m

        return (trajectory_stability + gait_stability) / 2.0

    def _determine_quality_rating(self, quality_score: float) -> MovementQuality:
        """Determine movement quality rating from score."""
        if quality_score >= 0.8:
            return MovementQuality.EXCELLENT
        elif quality_score >= 0.6:
            return MovementQuality.GOOD
        elif quality_score >= 0.4:
            return MovementQuality.FAIR
        else:
            return MovementQuality.POOR

    def _assess_stability(
        self, stability_score: float, gait_pattern: GaitPattern
    ) -> str:
        """Generate stability assessment text."""
        if stability_score >= 0.8:
            return (
                f"Excellent stability with {gait_pattern.stability_margin:.3f}m margin"
            )
        elif stability_score >= 0.6:
            return f"Good stability, margin of {gait_pattern.stability_margin:.3f}m"
        elif stability_score >= 0.4:
            return f"Fair stability, some instability detected"
        else:
            return f"Poor stability, frequent instabilities observed"

    def _assess_efficiency(
        self, efficiency_score: float, gait_pattern: GaitPattern
    ) -> str:
        """Generate efficiency assessment text."""
        if efficiency_score >= 0.8:
            return f"Highly efficient gait at {gait_pattern.get_walking_speed():.2f}m/s"
        elif efficiency_score >= 0.6:
            return (
                f"Good efficiency, walking at {gait_pattern.get_walking_speed():.2f}m/s"
            )
        elif efficiency_score >= 0.4:
            return f"Moderate efficiency, room for improvement"
        else:
            return f"Low efficiency, significant energy waste detected"

    def _generate_gait_recommendations(
        self,
        gait_pattern: GaitPattern,
        stability_score: float,
        efficiency_score: float,
        symmetry_score: float,
    ) -> List[str]:
        """Generate recommendations for gait improvement."""
        recommendations = []

        if stability_score < 0.6:
            recommendations.append(
                "Focus on balance training to improve stability margin"
            )
            recommendations.append("Reduce walking speed to improve stability")

        if efficiency_score < 0.6:
            recommendations.append(
                "Work on smoother joint movements to improve efficiency"
            )
            recommendations.append("Practice consistent stride patterns")

        if symmetry_score < 0.6:
            recommendations.append("Focus on symmetric gait training")
            recommendations.append("Practice equal weight distribution between legs")

        if gait_pattern.stride_frequency > 3.0:
            recommendations.append("Reduce stride frequency for more stable walking")
        elif gait_pattern.stride_frequency < 0.5:
            recommendations.append("Increase stride frequency for more dynamic walking")

        if not recommendations:
            recommendations.append(
                "Continue current training to maintain good gait quality"
            )

        return recommendations

    def _generate_efficiency_assessment(
        self, efficiency_score: float, distance: float, time: float
    ) -> str:
        """Generate efficiency assessment text."""
        speed = distance / time

        if efficiency_score >= 0.8:
            return f"Highly efficient movement at {speed:.2f}m/s"
        elif efficiency_score >= 0.6:
            return f"Good efficiency at {speed:.2f}m/s, minor optimization possible"
        elif efficiency_score >= 0.4:
            return f"Moderate efficiency at {speed:.2f}m/s, significant improvement possible"
        else:
            return f"Low efficiency at {speed:.2f}m/s, major optimization needed"

    def _analyze_movement_consistency(self, state_history: List[RobotState]) -> float:
        """Analyze consistency of movement patterns."""
        if len(state_history) < 5:
            return 0.0

        # Calculate velocity variations
        velocities = []
        for i in range(1, len(state_history)):
            prev_pos = state_history[i - 1].position
            curr_pos = state_history[i].position
            dt = 0.1  # Assume 10Hz
            velocity = np.linalg.norm(curr_pos - prev_pos) / dt
            velocities.append(velocity)

        if not velocities:
            return 0.0

        # Consistency = 1 - coefficient of variation
        mean_velocity = np.mean(velocities)
        if mean_velocity == 0:
            return 1.0  # Perfect consistency at zero velocity

        std_velocity = np.std(velocities)
        coefficient_of_variation = std_velocity / mean_velocity

        # Convert to 0-1 scale (lower variation = higher consistency)
        consistency = max(0.0, 1.0 - coefficient_of_variation)
        return min(consistency, 1.0)

    def _identify_primary_pattern(self, state_history: List[RobotState]) -> str:
        """Identify the primary movement pattern."""
        # Analyze position changes to identify pattern type
        positions = [state.position for state in state_history]

        # Calculate net displacement
        net_displacement = np.linalg.norm(positions[-1] - positions[0])

        # Analyze movement direction
        if net_displacement < 0.1:
            return "stationary_balance"

        # Check for forward movement
        forward_movement = positions[-1][0] - positions[0][0]
        if forward_movement > 0.5:
            return "forward_walking"
        elif forward_movement < -0.5:
            return "backward_walking"

        # Check for lateral movement
        lateral_movement = abs(positions[-1][1] - positions[0][1])
        if lateral_movement > 0.3:
            return "turning_movement"

        return "exploratory_movement"

    def _detect_learning_indicators(self, state_history: List[RobotState]) -> List[str]:
        """Detect indicators of learning progress."""
        indicators = []

        if len(state_history) < 10:
            return indicators

        # Analyze stability over time
        recent_states = state_history[-5:]
        early_states = state_history[:5]

        recent_heights = [state.position[2] for state in recent_states]
        early_heights = [state.position[2] for state in early_states]

        recent_height_var = np.var(recent_heights)
        early_height_var = np.var(early_heights)

        if recent_height_var < early_height_var * 0.8:
            indicators.append("Improved height stability")

        # Analyze movement smoothness
        recent_smoothness = self._calculate_trajectory_smoothness(recent_states)
        early_smoothness = self._calculate_trajectory_smoothness(early_states)

        if recent_smoothness > early_smoothness * 1.1:
            indicators.append("Smoother movement patterns")

        # Analyze consistency
        if len(indicators) >= 2:
            indicators.append("Multiple improvement areas detected")

        return indicators

    def _identify_improvement_areas(self, state_history: List[RobotState]) -> List[str]:
        """Identify areas needing improvement."""
        areas = []

        if len(state_history) < 5:
            areas.append("Need more movement data")
            return areas

        # Check height stability
        heights = [state.position[2] for state in state_history]
        height_var = np.var(heights)

        if height_var > 0.01:  # High height variation
            areas.append("Height stability")

        # Check forward progress
        forward_movement = state_history[-1].position[0] - state_history[0].position[0]
        if abs(forward_movement) < 0.1:
            areas.append("Forward locomotion")

        # Check joint position stability
        joint_vars = []
        for i in range(len(state_history[0].joint_positions)):
            joint_values = [state.joint_positions[i] for state in state_history]
            joint_vars.append(np.var(joint_values))

        if np.mean(joint_vars) > 0.1:
            areas.append("Joint coordination")

        return areas

    def _identify_movement_strengths(
        self, state_history: List[RobotState]
    ) -> List[str]:
        """Identify movement strengths."""
        strengths = []

        if len(state_history) < 5:
            return strengths

        # Check balance maintenance
        heights = [state.position[2] for state in state_history]
        if all(h > 0.5 for h in heights):  # Maintained upright
            strengths.append("Maintains upright posture")

        # Check forward progress
        forward_movement = state_history[-1].position[0] - state_history[0].position[0]
        if forward_movement > 0.5:
            strengths.append("Consistent forward progress")

        # Check stability
        height_var = np.var(heights)
        if height_var < 0.005:
            strengths.append("Stable height control")

        return strengths

    def _calculate_skill_metrics(
        self, skill_type: SkillType, trajectories: List[MovementTrajectory]
    ) -> Dict[str, float]:
        """Calculate skill-specific metrics."""
        if not trajectories:
            return {"overall_score": 0.0}

        metrics = {}

        # Common metrics for all skills
        total_distance = sum(traj.get_total_distance() for traj in trajectories)
        avg_distance = total_distance / len(trajectories)

        avg_velocity = sum(traj.get_average_velocity() for traj in trajectories) / len(
            trajectories
        )
        avg_smoothness = sum(
            traj.get_smoothness_score() for traj in trajectories
        ) / len(trajectories)

        metrics.update(
            {
                "avg_distance": avg_distance,
                "avg_velocity": avg_velocity,
                "avg_smoothness": avg_smoothness,
            }
        )

        # Skill-specific metrics
        if skill_type == SkillType.FORWARD_WALKING:
            # Focus on forward progress and consistency
            forward_consistency = self._calculate_forward_consistency(trajectories)
            metrics["forward_consistency"] = forward_consistency
            metrics["overall_score"] = (
                avg_smoothness + forward_consistency + min(avg_velocity / 1.0, 1.0)
            ) / 3.0

        elif skill_type == SkillType.STATIC_BALANCE:
            # Focus on minimal movement and stability
            movement_minimization = max(0.0, 1.0 - avg_distance)
            metrics["movement_minimization"] = movement_minimization
            metrics["overall_score"] = (avg_smoothness + movement_minimization) / 2.0

        elif skill_type == SkillType.TURNING:
            # Focus on rotational movement
            turning_quality = self._calculate_turning_quality(trajectories)
            metrics["turning_quality"] = turning_quality
            metrics["overall_score"] = (avg_smoothness + turning_quality) / 2.0

        else:
            # Default scoring
            metrics["overall_score"] = avg_smoothness

        return metrics

    def _calculate_progression_trend(self, skill_metrics: Dict[str, float]) -> str:
        """Calculate progression trend from skill metrics."""
        overall_score = skill_metrics.get("overall_score", 0.0)

        if overall_score >= 0.8:
            return "mastering"
        elif overall_score >= 0.6:
            return "improving"
        elif overall_score >= 0.4:
            return "developing"
        elif overall_score >= 0.2:
            return "learning"
        else:
            return "beginning"

    def _identify_mastery_indicators(
        self, skill_type: SkillType, skill_metrics: Dict[str, float]
    ) -> List[str]:
        """Identify indicators of skill mastery."""
        indicators = []

        overall_score = skill_metrics.get("overall_score", 0.0)

        if overall_score >= 0.7:
            indicators.append("High overall performance")

        if skill_metrics.get("avg_smoothness", 0.0) >= 0.8:
            indicators.append("Smooth movement execution")

        if skill_type == SkillType.FORWARD_WALKING:
            if skill_metrics.get("forward_consistency", 0.0) >= 0.7:
                indicators.append("Consistent forward locomotion")
            if skill_metrics.get("avg_velocity", 0.0) >= 0.8:
                indicators.append("Good walking speed")

        return indicators

    def _determine_next_milestones(
        self, skill_type: SkillType, skill_metrics: Dict[str, float]
    ) -> List[str]:
        """Determine next milestones for skill development."""
        milestones = []

        overall_score = skill_metrics.get("overall_score", 0.0)

        if overall_score < 0.3:
            milestones.append("Achieve basic skill execution")
        elif overall_score < 0.6:
            milestones.append("Improve consistency and smoothness")
        elif overall_score < 0.8:
            milestones.append("Refine technique for mastery")
        else:
            milestones.append("Maintain mastery and explore variations")

        # Skill-specific milestones
        if skill_type == SkillType.FORWARD_WALKING:
            if skill_metrics.get("avg_velocity", 0.0) < 0.5:
                milestones.append("Increase walking speed")
            if skill_metrics.get("forward_consistency", 0.0) < 0.6:
                milestones.append("Improve forward direction consistency")

        return milestones

    def _normalize_change(self, change: float, expected_range: float) -> float:
        """Normalize a change value to -1 to 1 scale."""
        return max(-1.0, min(1.0, change / expected_range))

    def _generate_improvement_summary(self, comparison: Dict[str, Any]) -> str:
        """Generate a summary of gait improvements."""
        improvement_score = comparison["overall_improvement"]

        if improvement_score > 0.5:
            return "Significant improvement in gait quality"
        elif improvement_score > 0.2:
            return "Moderate improvement in gait quality"
        elif improvement_score > -0.2:
            return "Stable gait quality with minor changes"
        elif improvement_score > -0.5:
            return "Some decline in gait quality"
        else:
            return "Significant decline in gait quality"

    def _calculate_trajectory_smoothness(self, states: List[RobotState]) -> float:
        """Calculate smoothness of trajectory from robot states."""
        if len(states) < 3:
            return 0.0

        positions = [state.position for state in states]

        # Calculate acceleration magnitudes
        accelerations = []
        for i in range(1, len(positions) - 1):
            # Simple finite difference acceleration
            dt = 0.1  # Assume 10Hz
            pos_prev = np.array(positions[i - 1])
            pos_curr = np.array(positions[i])
            pos_next = np.array(positions[i + 1])

            vel1 = (pos_curr - pos_prev) / dt
            vel2 = (pos_next - pos_curr) / dt
            accel = (vel2 - vel1) / dt

            accelerations.append(np.linalg.norm(accel))

        if not accelerations:
            return 0.0

        # Smoothness is inverse of average acceleration
        avg_acceleration = np.mean(accelerations)
        return 1.0 / (1.0 + avg_acceleration)

    def _calculate_forward_consistency(
        self, trajectories: List[MovementTrajectory]
    ) -> float:
        """Calculate consistency of forward movement."""
        if not trajectories:
            return 0.0

        forward_movements = []

        for trajectory in trajectories:
            if len(trajectory.positions) >= 2:
                start_pos = np.array(trajectory.positions[0])
                end_pos = np.array(trajectory.positions[-1])
                forward_movement = end_pos[0] - start_pos[0]  # X-axis movement
                forward_movements.append(forward_movement)

        if not forward_movements:
            return 0.0

        # Consistency = 1 - coefficient of variation
        mean_movement = np.mean(forward_movements)
        if mean_movement <= 0:
            return 0.0  # No forward movement

        std_movement = np.std(forward_movements)
        coefficient_of_variation = std_movement / mean_movement

        return max(0.0, 1.0 - coefficient_of_variation)

    def _calculate_turning_quality(
        self, trajectories: List[MovementTrajectory]
    ) -> float:
        """Calculate quality of turning movements."""
        if not trajectories:
            return 0.0

        turning_scores = []

        for trajectory in trajectories:
            if len(trajectory.positions) >= 3:
                # Calculate path curvature
                positions = [np.array(pos) for pos in trajectory.positions]
                curvatures = []

                for i in range(1, len(positions) - 1):
                    # Simple curvature calculation
                    p1, p2, p3 = positions[i - 1], positions[i], positions[i + 1]

                    v1 = p2 - p1
                    v2 = p3 - p2

                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (
                            np.linalg.norm(v1) * np.linalg.norm(v2)
                        )
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.arccos(cos_angle)
                        curvatures.append(angle)

                if curvatures:
                    # Quality based on consistent curvature
                    mean_curvature = np.mean(curvatures)
                    std_curvature = np.std(curvatures)

                    # Good turning has moderate, consistent curvature
                    curvature_quality = 1.0 / (1.0 + std_curvature)
                    turning_scores.append(curvature_quality)

        return np.mean(turning_scores) if turning_scores else 0.0

    def _calculate_velocity_consistency(self, trajectory: MovementTrajectory) -> float:
        """Calculate velocity consistency more accurately."""
        if len(trajectory.positions) < 3:
            return 0.0

        # Calculate step-by-step velocities from positions
        velocities = []
        for i in range(1, len(trajectory.positions)):
            pos_prev = np.array(trajectory.positions[i - 1])
            pos_curr = np.array(trajectory.positions[i])
            dt = (
                trajectory.timestamps[i] - trajectory.timestamps[i - 1]
                if len(trajectory.timestamps) > i
                else 0.1
            )
            if dt > 0:
                velocity = np.linalg.norm(pos_curr - pos_prev) / dt
                velocities.append(velocity)

        if len(velocities) < 2:
            return 0.0

        # Consistency = 1 - coefficient of variation
        mean_velocity = np.mean(velocities)
        if mean_velocity == 0:
            return 1.0  # Perfect consistency at zero velocity

        std_velocity = np.std(velocities)
        coefficient_of_variation = std_velocity / mean_velocity

        # Convert to 0-1 scale (lower variation = higher consistency)
        consistency = max(0.0, 1.0 - coefficient_of_variation)
        return min(consistency, 1.0)

    def _calculate_acceleration_smoothness(
        self, trajectory: MovementTrajectory
    ) -> float:
        """Calculate acceleration smoothness."""
        if len(trajectory.positions) < 3:
            return 0.0

        # Calculate accelerations
        accelerations = []
        for i in range(1, len(trajectory.positions) - 1):
            pos_prev = np.array(trajectory.positions[i - 1])
            pos_curr = np.array(trajectory.positions[i])
            pos_next = np.array(trajectory.positions[i + 1])

            dt = 0.1  # Assume fixed timestep
            vel1 = (pos_curr - pos_prev) / dt
            vel2 = (pos_next - pos_curr) / dt
            accel = (vel2 - vel1) / dt

            accelerations.append(np.linalg.norm(accel))

        if not accelerations:
            return 0.0

        # Smoothness is inverse of average acceleration magnitude
        avg_acceleration = np.mean(accelerations)
        return 1.0 / (1.0 + avg_acceleration)
