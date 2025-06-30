"""
Robot management endpoints for Genesis Humanoid RL API.

Provides endpoints for robot configuration, status monitoring, and skill assessment.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    RobotConfig,
    RobotStatus,
    RobotListResponse,
    SkillAssessmentRequest,
    SkillAssessmentResult,
    RobotTypeAPI,
    SkillTypeAPI,
    BaseResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demonstration
robots = {}
skill_assessments = {}

# Initialize with a default robot
default_robot_id = "unitree_g1_default"
robots[default_robot_id] = RobotStatus(
    robot_id=default_robot_id,
    robot_name="Unitree G1 Default",
    robot_type=RobotTypeAPI.UNITREE_G1,
    status="idle",
    learned_skills={
        SkillTypeAPI.POSTURAL_CONTROL: 0.7,
        SkillTypeAPI.STATIC_BALANCE: 0.6,
        SkillTypeAPI.FORWARD_WALKING: 0.4,
    },
    total_training_time=25.5,
    total_episodes=1250,
    last_activity=datetime.now(),
    performance_summary={
        "best_reward": 145.2,
        "average_reward": 98.7,
        "success_rate": 0.73,
    },
)


@router.get("/", response_model=RobotListResponse)
async def list_robots(
    robot_type: Optional[RobotTypeAPI] = None, status: Optional[str] = None
):
    """
    List all robots with optional filtering.

    Returns a list of all configured robots with their current status
    and performance information.
    """
    try:
        # Get all robots
        all_robots = list(robots.values())

        # Apply filters
        filtered_robots = all_robots

        if robot_type:
            filtered_robots = [r for r in filtered_robots if r.robot_type == robot_type]

        if status:
            filtered_robots = [r for r in filtered_robots if r.status == status]

        # Sort by last activity (most recent first)
        filtered_robots.sort(
            key=lambda x: x.last_activity or datetime.min, reverse=True
        )

        return RobotListResponse(
            robots=filtered_robots, total_count=len(filtered_robots)
        )

    except Exception as e:
        logger.error(f"Failed to list robots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list robots: {str(e)}")


@router.post("/", response_model=RobotStatus, status_code=201)
async def create_robot(config: RobotConfig):
    """
    Create a new robot configuration.

    Registers a new robot with the specified configuration.
    The robot will be available for training and evaluation.
    """
    try:
        robot_id = str(uuid.uuid4())

        robot = RobotStatus(
            robot_id=robot_id,
            robot_name=config.name,
            robot_type=config.robot_type,
            status="idle",
            learned_skills={},
            total_training_time=0.0,
            total_episodes=0,
            last_activity=datetime.now(),
            performance_summary={},
        )

        robots[robot_id] = robot

        logger.info(f"Created robot: {robot_id} ({config.name})")

        return robot

    except Exception as e:
        logger.error(f"Failed to create robot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create robot: {str(e)}")


@router.get("/{robot_id}", response_model=RobotStatus)
async def get_robot(robot_id: str):
    """
    Get a specific robot by ID.

    Returns detailed information about the specified robot.
    """
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")

    return robots[robot_id]


@router.put("/{robot_id}", response_model=RobotStatus)
async def update_robot(robot_id: str, config: RobotConfig):
    """
    Update a robot configuration.

    Updates the configuration of an existing robot.
    Robot must not be in use (training or evaluating).
    """
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")

    robot = robots[robot_id]

    if robot.status not in ["idle", "error"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update robot while in {robot.status} status",
        )

    try:
        # Update robot configuration
        robot.robot_name = config.name
        robot.robot_type = config.robot_type
        robot.last_activity = datetime.now()

        robots[robot_id] = robot

        logger.info(f"Updated robot: {robot_id}")

        return robot

    except Exception as e:
        logger.error(f"Failed to update robot {robot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update robot: {str(e)}")


@router.delete("/{robot_id}", response_model=BaseResponse)
async def delete_robot(robot_id: str):
    """
    Delete a robot.

    Removes a robot configuration and associated data.
    Robot must not be in use.
    """
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")

    robot = robots[robot_id]

    if robot.status not in ["idle", "error"]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete robot while in {robot.status} status",
        )

    try:
        # Remove robot and associated assessments
        del robots[robot_id]

        # Remove skill assessments for this robot
        assessments_to_remove = [
            aid
            for aid, assessment in skill_assessments.items()
            if assessment.robot_id == robot_id
        ]

        for aid in assessments_to_remove:
            del skill_assessments[aid]

        logger.info(f"Deleted robot: {robot_id}")

        return BaseResponse(
            success=True, message=f"Robot {robot_id} deleted successfully"
        )

    except Exception as e:
        logger.error(f"Failed to delete robot {robot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete robot: {str(e)}")


@router.post(
    "/{robot_id}/assess-skills",
    response_model=List[SkillAssessmentResult],
    status_code=201,
)
async def assess_robot_skills(
    robot_id: str, request: SkillAssessmentRequest, background_tasks: BackgroundTasks
):
    """
    Assess robot skills.

    Evaluates the robot's proficiency in specified skills through
    dedicated assessment episodes.
    """
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")

    robot = robots[robot_id]

    if robot.status != "idle":
        raise HTTPException(
            status_code=400,
            detail=f"Cannot assess skills while robot is in {robot.status} status",
        )

    try:
        assessment_results = []

        for skill_type in request.skill_types:
            assessment_id = str(uuid.uuid4())

            assessment = SkillAssessmentResult(
                assessment_id=assessment_id,
                robot_id=robot_id,
                skill_type=skill_type,
                proficiency_score=0.0,  # Will be updated by background task
                confidence_level=0.0,
                evidence_quality=0.0,
                assessment_episodes=request.assessment_episodes,
                success_rate=0.0,
                average_reward=0.0,
                assessed_at=datetime.now(),
            )

            skill_assessments[assessment_id] = assessment
            assessment_results.append(assessment)

            # Start assessment in background
            background_tasks.add_task(
                simulate_skill_assessment, assessment_id, robot_id, skill_type, request
            )

        # Update robot status
        robot.status = "evaluating"
        robot.current_session_id = (
            f"skill_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        robot.last_activity = datetime.now()
        robots[robot_id] = robot

        logger.info(
            f"Started skill assessment for robot {robot_id}: {len(request.skill_types)} skills"
        )

        return assessment_results

    except Exception as e:
        logger.error(f"Failed to assess robot skills for {robot_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to assess robot skills: {str(e)}"
        )


@router.get("/{robot_id}/skills", response_model=List[SkillAssessmentResult])
async def get_robot_skill_assessments(
    robot_id: str,
    skill_type: Optional[SkillTypeAPI] = None,
    limit: int = Query(50, ge=1, le=1000),
):
    """
    Get skill assessments for a robot.

    Returns recent skill assessment results for the specified robot.
    """
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")

    try:
        # Get assessments for this robot
        robot_assessments = [
            assessment
            for assessment in skill_assessments.values()
            if assessment.robot_id == robot_id
        ]

        # Filter by skill type if specified
        if skill_type:
            robot_assessments = [
                assessment
                for assessment in robot_assessments
                if assessment.skill_type == skill_type
            ]

        # Sort by assessment date (most recent first)
        robot_assessments.sort(key=lambda x: x.assessed_at, reverse=True)

        # Apply limit
        robot_assessments = robot_assessments[:limit]

        return robot_assessments

    except Exception as e:
        logger.error(f"Failed to get skill assessments for robot {robot_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get robot skill assessments: {str(e)}"
        )


@router.get("/{robot_id}/performance")
async def get_robot_performance(robot_id: str):
    """
    Get robot performance analytics.

    Returns detailed performance analytics and learning progress
    for the specified robot.
    """
    if robot_id not in robots:
        raise HTTPException(status_code=404, detail=f"Robot {robot_id} not found")

    robot = robots[robot_id]

    try:
        # Generate performance analytics
        performance_data = {
            "robot_info": {
                "robot_id": robot.robot_id,
                "robot_name": robot.robot_name,
                "robot_type": robot.robot_type.value,
                "total_training_time": robot.total_training_time,
                "total_episodes": robot.total_episodes,
            },
            "skill_proficiency": robot.learned_skills,
            "performance_metrics": robot.performance_summary,
            "learning_progress": {
                "skill_acquisition_rate": 0.15,  # Skills per hour
                "improvement_trend": "increasing",
                "learning_efficiency": 0.75,
                "stability_score": 0.82,
            },
            "recent_activity": {
                "last_training_session": (
                    robot.last_activity.isoformat() if robot.last_activity else None
                ),
                "sessions_this_week": 3,
                "average_session_duration": 2.5,
                "recent_success_rate": 0.78,
            },
            "skill_breakdown": [
                {
                    "skill": skill.value,
                    "proficiency": proficiency,
                    "confidence": 0.85,
                    "last_assessed": datetime.now().isoformat(),
                    "improvement_trend": "stable",
                }
                for skill, proficiency in robot.learned_skills.items()
            ],
        }

        return JSONResponse(content=performance_data)

    except Exception as e:
        logger.error(f"Failed to get performance data for robot {robot_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get robot performance data: {str(e)}"
        )


async def simulate_skill_assessment(
    assessment_id: str,
    robot_id: str,
    skill_type: SkillTypeAPI,
    request: SkillAssessmentRequest,
):
    """
    Background task to simulate skill assessment.

    In a real implementation, this would run assessment episodes
    in the Genesis environment to evaluate the robot's skill level.
    """
    import asyncio
    import random

    try:
        assessment = skill_assessments.get(assessment_id)
        if not assessment:
            return

        logger.info(f"Starting skill assessment {assessment_id} for {skill_type.value}")

        # Simulate assessment episodes
        episode_rewards = []
        successful_episodes = 0

        for episode in range(request.assessment_episodes):
            # Simulate episode assessment
            await asyncio.sleep(0.5)  # Simulate episode time

            # Generate skill-specific results
            base_proficiency = {
                SkillTypeAPI.POSTURAL_CONTROL: 0.8,
                SkillTypeAPI.STATIC_BALANCE: 0.7,
                SkillTypeAPI.DYNAMIC_BALANCE: 0.6,
                SkillTypeAPI.FORWARD_WALKING: 0.5,
                SkillTypeAPI.BACKWARD_WALKING: 0.4,
                SkillTypeAPI.TURNING: 0.4,
                SkillTypeAPI.SPEED_CONTROL: 0.3,
                SkillTypeAPI.TERRAIN_ADAPTATION: 0.2,
                SkillTypeAPI.OBSTACLE_AVOIDANCE: 0.2,
            }.get(skill_type, 0.5)

            # Add noise to base proficiency
            episode_reward = random.gauss(base_proficiency * 100, 20)
            episode_reward = max(0, min(100, episode_reward))  # Clamp to 0-100

            episode_rewards.append(episode_reward)

            if episode_reward > 60:  # Success threshold
                successful_episodes += 1

        # Calculate final assessment metrics
        average_reward = sum(episode_rewards) / len(episode_rewards)
        success_rate = successful_episodes / len(episode_rewards)
        proficiency_score = min(1.0, average_reward / 100)

        # Calculate confidence and evidence quality based on consistency
        reward_std = sum(
            (r - average_reward) ** 2 for r in episode_rewards
        ) ** 0.5 / len(episode_rewards)
        confidence_level = max(
            0.3, 1.0 - (reward_std / 50)
        )  # Higher confidence for consistent results
        evidence_quality = min(
            1.0, len(episode_rewards) / 20
        )  # More episodes = better evidence

        # Generate recommendations
        recommendations = []
        if proficiency_score < 0.5:
            recommendations.append(
                f"Requires significant training in {skill_type.value}"
            )
        elif proficiency_score < 0.7:
            recommendations.append(
                f"Moderate proficiency in {skill_type.value}, continue training"
            )
        else:
            recommendations.append(
                f"Good proficiency in {skill_type.value}, ready for advanced training"
            )

        if confidence_level < 0.7:
            recommendations.append(
                "Consider more assessment episodes for better confidence"
            )

        # Update assessment results
        assessment.proficiency_score = proficiency_score
        assessment.confidence_level = confidence_level
        assessment.evidence_quality = evidence_quality
        assessment.success_rate = success_rate
        assessment.average_reward = average_reward
        assessment.recommendations = recommendations

        skill_assessments[assessment_id] = assessment

        # Update robot's learned skills
        robot = robots.get(robot_id)
        if robot:
            robot.learned_skills[skill_type] = proficiency_score
            robot.last_activity = datetime.now()

            # Check if all assessments for this robot are complete
            robot_assessments = [
                a
                for a in skill_assessments.values()
                if a.robot_id == robot_id
                and a.assessed_at.date() == datetime.now().date()
            ]

            if all(a.proficiency_score > 0 for a in robot_assessments):
                robot.status = "idle"
                robot.current_session_id = None

            robots[robot_id] = robot

        logger.info(
            f"Completed skill assessment {assessment_id}: {skill_type.value} = {proficiency_score:.2f}"
        )

    except Exception as e:
        logger.error(f"Skill assessment simulation failed for {assessment_id}: {e}")

        # Mark assessment as failed
        assessment = skill_assessments.get(assessment_id)
        if assessment:
            assessment.recommendations = [f"Assessment failed: {str(e)}"]
            skill_assessments[assessment_id] = assessment

        # Reset robot status
        robot = robots.get(robot_id)
        if robot:
            robot.status = "error"
            robot.current_session_id = None
            robots[robot_id] = robot
