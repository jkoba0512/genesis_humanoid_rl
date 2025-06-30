"""
Training management endpoints for Genesis Humanoid RL API.

Provides endpoints for creating, managing, and monitoring training sessions.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    CreateTrainingRequest,
    UpdateTrainingRequest,
    TrainingControlRequest,
    TrainingSessionResponse,
    TrainingSessionListResponse,
    TrainingMetrics,
    MetricsResponse,
    TrainingStatus,
    PaginationParams,
    FilterParams,
    BaseResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demonstration
# In production, this would be a proper database
training_sessions = {}
training_metrics = {}


@router.post("/sessions", response_model=TrainingSessionResponse, status_code=201)
async def create_training_session(request: CreateTrainingRequest):
    """
    Create a new training session.

    Creates a new training session with the specified configuration.
    The session will be in 'created' status and ready to start.
    """
    try:
        session_id = str(uuid.uuid4())

        session = TrainingSessionResponse(
            session_id=session_id,
            session_name=request.session_name,
            status=TrainingStatus.CREATED,
            robot_config=request.robot_config,
            training_config=request.training_config,
            environment_config=request.environment_config,
            curriculum_config=request.curriculum_config,
            description=request.description,
            tags=request.tags,
            created_at=datetime.now(),
            progress={
                "timesteps_completed": 0,
                "episodes_completed": 0,
                "success_rate": 0.0,
                "current_reward": 0.0,
            },
            metrics={
                "best_reward": 0.0,
                "average_reward": 0.0,
                "training_fps": 0.0,
                "estimated_time_remaining": 0.0,
            },
        )

        training_sessions[session_id] = session
        training_metrics[session_id] = []

        logger.info(f"Created training session: {session_id}")

        return session

    except Exception as e:
        logger.error(f"Failed to create training session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create training session: {str(e)}"
        )


@router.get("/sessions", response_model=TrainingSessionListResponse)
async def list_training_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    status: Optional[TrainingStatus] = None,
    robot_type: Optional[str] = None,
    search: Optional[str] = None,
):
    """
    List training sessions with filtering and pagination.

    Returns a paginated list of training sessions with optional filtering
    by status, robot type, or search terms.
    """
    try:
        # Get all sessions
        all_sessions = list(training_sessions.values())

        # Apply filters
        filtered_sessions = all_sessions

        if status:
            filtered_sessions = [s for s in filtered_sessions if s.status == status]

        if robot_type:
            filtered_sessions = [
                s
                for s in filtered_sessions
                if s.robot_config.robot_type.value == robot_type
            ]

        if search:
            search_lower = search.lower()
            filtered_sessions = [
                s
                for s in filtered_sessions
                if search_lower in s.session_name.lower()
                or (s.description and search_lower in s.description.lower())
                or any(search_lower in tag.lower() for tag in s.tags)
            ]

        # Calculate pagination
        total_count = len(filtered_sessions)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paginated_sessions = filtered_sessions[start_idx:end_idx]

        return TrainingSessionListResponse(
            sessions=paginated_sessions,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(f"Failed to list training sessions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list training sessions: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=TrainingSessionResponse)
async def get_training_session(session_id: str):
    """
    Get a specific training session by ID.

    Returns detailed information about the specified training session.
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404, detail=f"Training session {session_id} not found"
        )

    return training_sessions[session_id]


@router.put("/sessions/{session_id}", response_model=TrainingSessionResponse)
async def update_training_session(session_id: str, request: UpdateTrainingRequest):
    """
    Update a training session configuration.

    Updates the configuration of an existing training session.
    Some parameters can only be updated when the session is not running.
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404, detail=f"Training session {session_id} not found"
        )

    session = training_sessions[session_id]

    # Check if session can be updated
    if session.status == TrainingStatus.RUNNING:
        # Only allow limited updates while running
        if request.training_config or request.environment_config:
            raise HTTPException(
                status_code=400,
                detail="Cannot update training or environment config while session is running",
            )

    try:
        # Update allowed fields
        if request.session_name:
            session.session_name = request.session_name

        if request.description:
            session.description = request.description

        if request.tags is not None:
            session.tags = request.tags

        if request.training_config and session.status != TrainingStatus.RUNNING:
            session.training_config = request.training_config

        if request.environment_config and session.status != TrainingStatus.RUNNING:
            session.environment_config = request.environment_config

        training_sessions[session_id] = session

        logger.info(f"Updated training session: {session_id}")

        return session

    except Exception as e:
        logger.error(f"Failed to update training session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to update training session: {str(e)}"
        )


@router.post("/sessions/{session_id}/control", response_model=BaseResponse)
async def control_training_session(
    session_id: str, request: TrainingControlRequest, background_tasks: BackgroundTasks
):
    """
    Control a training session (start, pause, resume, stop, cancel).

    Manages the execution state of a training session.
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404, detail=f"Training session {session_id} not found"
        )

    session = training_sessions[session_id]
    current_status = session.status

    try:
        if request.action == "start":
            if current_status != TrainingStatus.CREATED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot start session in {current_status} status",
                )

            session.status = TrainingStatus.RUNNING
            session.started_at = datetime.now()

            # Start training in background
            background_tasks.add_task(simulate_training, session_id)

            message = f"Training session {session_id} started"

        elif request.action == "pause":
            if current_status != TrainingStatus.RUNNING:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot pause session in {current_status} status",
                )

            session.status = TrainingStatus.PAUSED
            message = f"Training session {session_id} paused"

        elif request.action == "resume":
            if current_status != TrainingStatus.PAUSED:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot resume session in {current_status} status",
                )

            session.status = TrainingStatus.RUNNING

            # Resume training in background
            background_tasks.add_task(simulate_training, session_id)

            message = f"Training session {session_id} resumed"

        elif request.action == "stop":
            if current_status not in [TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot stop session in {current_status} status",
                )

            session.status = TrainingStatus.COMPLETED
            session.completed_at = datetime.now()
            message = f"Training session {session_id} stopped"

        elif request.action == "cancel":
            if current_status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot cancel session in {current_status} status",
                )

            session.status = TrainingStatus.CANCELLED
            session.completed_at = datetime.now()
            message = f"Training session {session_id} cancelled"

        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown action: {request.action}"
            )

        training_sessions[session_id] = session

        logger.info(
            f"Training control action '{request.action}' for session {session_id}"
        )

        return BaseResponse(success=True, message=message)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to control training session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to control training session: {str(e)}"
        )


@router.delete("/sessions/{session_id}", response_model=BaseResponse)
async def delete_training_session(session_id: str):
    """
    Delete a training session.

    Removes a training session and associated data.
    Only stopped or failed sessions can be deleted.
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404, detail=f"Training session {session_id} not found"
        )

    session = training_sessions[session_id]

    if session.status in [TrainingStatus.RUNNING, TrainingStatus.PAUSED]:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete running or paused session. Stop the session first.",
        )

    try:
        # Remove session and metrics
        del training_sessions[session_id]
        if session_id in training_metrics:
            del training_metrics[session_id]

        logger.info(f"Deleted training session: {session_id}")

        return BaseResponse(
            success=True, message=f"Training session {session_id} deleted successfully"
        )

    except Exception as e:
        logger.error(f"Failed to delete training session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete training session: {str(e)}"
        )


@router.get("/sessions/{session_id}/metrics", response_model=MetricsResponse)
async def get_training_metrics(
    session_id: str,
    limit: int = Query(1000, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """
    Get training metrics for a session.

    Returns time-series training metrics including rewards, losses,
    and performance indicators.
    """
    if session_id not in training_sessions:
        raise HTTPException(
            status_code=404, detail=f"Training session {session_id} not found"
        )

    try:
        session_metrics = training_metrics.get(session_id, [])

        # Apply pagination
        total_count = len(session_metrics)
        paginated_metrics = session_metrics[offset : offset + limit]

        # Get time range
        start_time = None
        end_time = None
        if paginated_metrics:
            start_time = paginated_metrics[0].timestamp
            end_time = paginated_metrics[-1].timestamp

        return MetricsResponse(
            metrics=paginated_metrics,
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
        )

    except Exception as e:
        logger.error(f"Failed to get metrics for session {session_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get training metrics: {str(e)}"
        )


async def simulate_training(session_id: str):
    """
    Background task to simulate training progress.

    In a real implementation, this would interface with the actual
    training system (Stable-Baselines3, etc.).
    """
    import asyncio
    import random

    try:
        session = training_sessions.get(session_id)
        if not session:
            return

        total_timesteps = session.training_config.total_timesteps
        current_timesteps = 0
        episode = 0

        while (
            session.status == TrainingStatus.RUNNING
            and current_timesteps < total_timesteps
        ):

            # Simulate training step
            await asyncio.sleep(1)  # Simulate computation time

            current_timesteps += 1000  # Simulate progress
            episode += 1

            # Generate simulated metrics
            episode_reward = (
                random.uniform(50, 150) + (current_timesteps / total_timesteps) * 100
            )
            episode_length = random.randint(80, 120)

            metric = TrainingMetrics(
                episode=episode,
                timestep=current_timesteps,
                episode_reward=episode_reward,
                episode_length=episode_length,
                success_rate=min(0.9, current_timesteps / total_timesteps * 0.8),
                fps=random.uniform(45, 65),
                learning_rate=session.training_config.learning_rate
                * (0.99 ** (current_timesteps / 10000)),
                policy_loss=random.uniform(0.01, 0.1),
                value_loss=random.uniform(0.05, 0.2),
                entropy_loss=random.uniform(0.001, 0.01),
                explained_variance=random.uniform(0.6, 0.9),
            )

            # Store metrics
            if session_id not in training_metrics:
                training_metrics[session_id] = []
            training_metrics[session_id].append(metric)

            # Update session progress
            session.progress.update(
                {
                    "timesteps_completed": current_timesteps,
                    "episodes_completed": episode,
                    "success_rate": metric.success_rate,
                    "current_reward": episode_reward,
                }
            )

            session.metrics.update(
                {
                    "best_reward": max(
                        session.metrics.get("best_reward", 0), episode_reward
                    ),
                    "average_reward": sum(
                        m.episode_reward for m in training_metrics[session_id][-100:]
                    )
                    / min(100, len(training_metrics[session_id])),
                    "training_fps": metric.fps,
                    "estimated_time_remaining": (total_timesteps - current_timesteps)
                    / 1000,
                }
            )

            # Refresh session status
            session = training_sessions.get(session_id)
            if not session or session.status != TrainingStatus.RUNNING:
                break

        # Training completed
        if session and current_timesteps >= total_timesteps:
            session.status = TrainingStatus.COMPLETED
            session.completed_at = datetime.now()
            session.model_path = f"./models/{session_id}/final_model.zip"
            training_sessions[session_id] = session

            logger.info(f"Training session {session_id} completed")

    except Exception as e:
        logger.error(f"Training simulation failed for session {session_id}: {e}")

        # Mark session as failed
        session = training_sessions.get(session_id)
        if session:
            session.status = TrainingStatus.FAILED
            session.error_message = str(e)
            session.completed_at = datetime.now()
            training_sessions[session_id] = session
