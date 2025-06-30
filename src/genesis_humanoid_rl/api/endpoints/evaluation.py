"""
Evaluation endpoints for Genesis Humanoid RL API.

Provides endpoints for model evaluation, performance assessment, and result analysis.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse

from ..models import (
    EvaluationRequest,
    EvaluationResult,
    EvaluationListResponse,
    EvaluationStatus,
    BaseResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demonstration
evaluations = {}


@router.post("/evaluate", response_model=EvaluationResult, status_code=201)
async def create_evaluation(
    request: EvaluationRequest, background_tasks: BackgroundTasks
):
    """
    Create a new model evaluation.

    Starts evaluation of a trained model with specified parameters.
    The evaluation runs in the background and results are available via polling.
    """
    try:
        evaluation_id = str(uuid.uuid4())

        evaluation = EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_name=request.evaluation_name,
            model_path=request.model_path,
            status=EvaluationStatus.PENDING,
            num_episodes=request.num_episodes,
            video_path=request.video_path if request.save_video else None,
            created_at=datetime.now(),
        )

        evaluations[evaluation_id] = evaluation

        # Start evaluation in background
        background_tasks.add_task(simulate_evaluation, evaluation_id, request)

        logger.info(f"Created evaluation: {evaluation_id}")

        return evaluation

    except Exception as e:
        logger.error(f"Failed to create evaluation: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to create evaluation: {str(e)}"
        )


@router.get("/evaluations", response_model=EvaluationListResponse)
async def list_evaluations(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=1000),
    status: Optional[EvaluationStatus] = None,
    model_path: Optional[str] = None,
):
    """
    List evaluations with filtering and pagination.

    Returns a paginated list of evaluations with optional filtering.
    """
    try:
        # Get all evaluations
        all_evaluations = list(evaluations.values())

        # Apply filters
        filtered_evaluations = all_evaluations

        if status:
            filtered_evaluations = [
                e for e in filtered_evaluations if e.status == status
            ]

        if model_path:
            filtered_evaluations = [
                e for e in filtered_evaluations if model_path in e.model_path
            ]

        # Sort by creation time (newest first)
        filtered_evaluations.sort(key=lambda x: x.created_at, reverse=True)

        # Calculate pagination
        total_count = len(filtered_evaluations)
        total_pages = (total_count + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size

        paginated_evaluations = filtered_evaluations[start_idx:end_idx]

        return EvaluationListResponse(
            evaluations=paginated_evaluations,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    except Exception as e:
        logger.error(f"Failed to list evaluations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to list evaluations: {str(e)}"
        )


@router.get("/evaluations/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation(evaluation_id: str):
    """
    Get a specific evaluation by ID.

    Returns detailed information about the specified evaluation.
    """
    if evaluation_id not in evaluations:
        raise HTTPException(
            status_code=404, detail=f"Evaluation {evaluation_id} not found"
        )

    return evaluations[evaluation_id]


@router.delete("/evaluations/{evaluation_id}", response_model=BaseResponse)
async def delete_evaluation(evaluation_id: str):
    """
    Delete an evaluation.

    Removes an evaluation and associated data.
    Running evaluations will be cancelled.
    """
    if evaluation_id not in evaluations:
        raise HTTPException(
            status_code=404, detail=f"Evaluation {evaluation_id} not found"
        )

    try:
        evaluation = evaluations[evaluation_id]

        # Cancel if running
        if evaluation.status == EvaluationStatus.RUNNING:
            evaluation.status = EvaluationStatus.FAILED
            evaluation.error_message = "Cancelled by user"
            evaluation.completed_at = datetime.now()

        # Remove evaluation
        del evaluations[evaluation_id]

        logger.info(f"Deleted evaluation: {evaluation_id}")

        return BaseResponse(
            success=True, message=f"Evaluation {evaluation_id} deleted successfully"
        )

    except Exception as e:
        logger.error(f"Failed to delete evaluation {evaluation_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to delete evaluation: {str(e)}"
        )


@router.post("/evaluations/{evaluation_id}/cancel", response_model=BaseResponse)
async def cancel_evaluation(evaluation_id: str):
    """
    Cancel a running evaluation.

    Stops a running evaluation and marks it as cancelled.
    """
    if evaluation_id not in evaluations:
        raise HTTPException(
            status_code=404, detail=f"Evaluation {evaluation_id} not found"
        )

    evaluation = evaluations[evaluation_id]

    if evaluation.status not in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel evaluation in {evaluation.status} status",
        )

    try:
        evaluation.status = EvaluationStatus.FAILED
        evaluation.error_message = "Cancelled by user"
        evaluation.completed_at = datetime.now()

        evaluations[evaluation_id] = evaluation

        logger.info(f"Cancelled evaluation: {evaluation_id}")

        return BaseResponse(
            success=True, message=f"Evaluation {evaluation_id} cancelled successfully"
        )

    except Exception as e:
        logger.error(f"Failed to cancel evaluation {evaluation_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cancel evaluation: {str(e)}"
        )


async def simulate_evaluation(evaluation_id: str, request: EvaluationRequest):
    """
    Background task to simulate evaluation.

    In a real implementation, this would load the model and run
    evaluation episodes in the Genesis environment.
    """
    import asyncio
    import random

    try:
        evaluation = evaluations.get(evaluation_id)
        if not evaluation:
            return

        # Start evaluation
        evaluation.status = EvaluationStatus.RUNNING
        evaluations[evaluation_id] = evaluation

        logger.info(f"Starting evaluation {evaluation_id}")

        episode_results = []
        total_reward = 0.0
        successful_episodes = 0

        for episode in range(request.num_episodes):
            # Check if evaluation was cancelled
            current_eval = evaluations.get(evaluation_id)
            if not current_eval or current_eval.status != EvaluationStatus.RUNNING:
                logger.info(f"Evaluation {evaluation_id} was cancelled")
                return

            # Simulate episode evaluation
            await asyncio.sleep(2)  # Simulate episode time

            # Generate simulated results
            episode_reward = random.uniform(80, 180)
            episode_length = random.randint(150, 300)
            success = episode_reward > 120  # Arbitrary success threshold

            episode_result = {
                "episode": episode + 1,
                "reward": episode_reward,
                "length": episode_length,
                "success": success,
                "completion_time": random.uniform(10, 25),
            }

            episode_results.append(episode_result)
            total_reward += episode_reward
            if success:
                successful_episodes += 1

            # Update progress
            evaluation.episodes_completed = episode + 1
            evaluation.detailed_results = episode_results
            evaluations[evaluation_id] = evaluation

        # Calculate final metrics
        evaluation.average_reward = total_reward / request.num_episodes
        evaluation.success_rate = successful_episodes / request.num_episodes
        evaluation.average_episode_length = sum(
            r["length"] for r in episode_results
        ) / len(episode_results)

        # Complete evaluation
        evaluation.status = EvaluationStatus.COMPLETED
        evaluation.completed_at = datetime.now()

        # Generate video path if requested
        if request.save_video:
            evaluation.video_path = f"./videos/{evaluation_id}/evaluation_video.mp4"

        evaluations[evaluation_id] = evaluation

        logger.info(
            f"Completed evaluation {evaluation_id}: avg_reward={evaluation.average_reward:.2f}, success_rate={evaluation.success_rate:.2%}"
        )

    except Exception as e:
        logger.error(f"Evaluation simulation failed for {evaluation_id}: {e}")

        # Mark evaluation as failed
        evaluation = evaluations.get(evaluation_id)
        if evaluation:
            evaluation.status = EvaluationStatus.FAILED
            evaluation.error_message = str(e)
            evaluation.completed_at = datetime.now()
            evaluations[evaluation_id] = evaluation
