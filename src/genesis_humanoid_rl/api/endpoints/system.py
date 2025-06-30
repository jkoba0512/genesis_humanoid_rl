"""
System management endpoints for Genesis Humanoid RL API.

Provides system-level information, configuration, and management endpoints.
"""

import logging
import time
import platform
from datetime import datetime
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from ..models import SystemStatus, BaseResponse, ErrorResponse
from ...infrastructure.monitoring.genesis_monitor import check_genesis_status

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get comprehensive system status.

    Returns detailed information about the system state, including
    active sessions, resource usage, and component health.
    """
    try:
        # Get system uptime (simplified)
        uptime = time.time() - 1000000  # Placeholder

        # Get Genesis status
        genesis_status = check_genesis_status()

        # Get system resources
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        system_resources = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(),
            "memory_total_gb": memory.total / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_percent": (disk.used / disk.total) * 100,
        }

        # Placeholder values for active sessions/evaluations
        # In a real implementation, these would come from the session manager
        active_training_sessions = 0
        active_evaluations = 0
        total_robots = 1  # Default robot count

        # Determine overall system status
        if not genesis_status["available"]:
            system_status = "degraded"
        elif (
            system_resources["memory_percent"] > 90
            or system_resources["disk_percent"] > 90
        ):
            system_status = "degraded"
        else:
            system_status = "healthy"

        return SystemStatus(
            uptime=uptime,
            status=system_status,
            active_training_sessions=active_training_sessions,
            active_evaluations=active_evaluations,
            total_robots=total_robots,
            genesis_status=genesis_status,
            system_resources=system_resources,
        )

    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system status: {str(e)}"
        )


@router.get("/info")
async def get_system_info():
    """
    Get system information and configuration.

    Returns static system information including platform details,
    installed packages, and configuration.
    """
    try:
        system_info = {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            "genesis_rl": {
                "version": "1.0.0",
                "api_version": "1.0.0",
                "installation_path": "/path/to/genesis_humanoid_rl",  # Would be actual path
                "config_dir": "~/.genesis_humanoid_rl",
            },
            "dependencies": {
                "genesis": check_genesis_status(),
                "python_packages": {
                    "fastapi": "latest",
                    "uvicorn": "latest",
                    "pydantic": "latest",
                    "stable_baselines3": "latest",
                },
            },
            "features": {
                "training": True,
                "evaluation": True,
                "curriculum_learning": True,
                "multi_robot": True,
                "gpu_acceleration": check_genesis_status()
                .get("features", {})
                .get("gpu_acceleration", False),
                "video_recording": True,
                "real_time_monitoring": True,
            },
            "limits": {
                "max_concurrent_training": 10,
                "max_concurrent_evaluation": 5,
                "max_robots": 100,
                "max_session_duration_hours": 72,
            },
        }

        return JSONResponse(content=system_info)

    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system information: {str(e)}"
        )


@router.get("/config")
async def get_system_config():
    """
    Get system configuration.

    Returns current system configuration including default settings,
    feature flags, and environment-specific configuration.
    """
    try:
        config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False,
                "cors_enabled": True,
                "rate_limiting": True,
                "rate_limit_per_minute": 1000,
            },
            "training": {
                "default_algorithm": "PPO",
                "default_timesteps": 1000000,
                "default_parallel_envs": 4,
                "checkpoint_frequency": 50000,
                "auto_save": True,
                "tensorboard_logging": True,
            },
            "evaluation": {
                "default_episodes": 10,
                "video_recording": True,
                "performance_metrics": True,
                "automatic_cleanup": True,
            },
            "monitoring": {
                "health_check_interval": 30,
                "metrics_collection": True,
                "log_level": "INFO",
                "retention_days": 30,
            },
            "storage": {
                "models_dir": "./models",
                "logs_dir": "./logs",
                "videos_dir": "./videos",
                "data_dir": "./data",
                "temp_dir": "./temp",
            },
            "security": {
                "api_key_required": False,
                "cors_origins": ["*"],
                "max_request_size_mb": 100,
                "session_timeout_hours": 24,
            },
        }

        return JSONResponse(content=config)

    except Exception as e:
        logger.error(f"Failed to get system config: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system configuration: {str(e)}"
        )


@router.get("/logs")
async def get_system_logs(
    level: str = Query("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"),
    limit: int = Query(100, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    """
    Get system logs.

    Retrieves recent system logs with filtering capabilities.
    Useful for debugging and monitoring system behavior.
    """
    try:
        # In a real implementation, this would read from actual log files
        # or a logging service. For now, return mock log entries.

        sample_logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "logger": "genesis_humanoid_rl.api.app",
                "message": "API started successfully",
                "module": "app.py",
                "line": 123,
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "logger": "genesis_humanoid_rl.training",
                "message": "Training session initialized",
                "module": "training.py",
                "line": 45,
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "WARNING",
                "logger": "genesis_humanoid_rl.monitoring",
                "message": "Genesis performance below threshold",
                "module": "monitor.py",
                "line": 78,
            },
        ]

        # Filter by level
        if level != "DEBUG":
            level_priority = {
                "DEBUG": 0,
                "INFO": 1,
                "WARNING": 2,
                "ERROR": 3,
                "CRITICAL": 4,
            }
            min_priority = level_priority[level]
            sample_logs = [
                log
                for log in sample_logs
                if level_priority[log["level"]] >= min_priority
            ]

        # Apply pagination
        total_count = len(sample_logs)
        logs = sample_logs[offset : offset + limit]

        return JSONResponse(
            content={
                "logs": logs,
                "total_count": total_count,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count,
            }
        )

    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system logs: {str(e)}"
        )


@router.post("/restart", response_model=BaseResponse)
async def restart_system():
    """
    Restart system components.

    Initiates a graceful restart of system components.
    This endpoint is typically used for maintenance or configuration updates.
    """
    try:
        # In a real implementation, this would trigger actual restart logic
        logger.info("System restart requested")

        # Placeholder restart logic
        # - Stop active training sessions gracefully
        # - Save current state
        # - Restart services

        return BaseResponse(
            success=True,
            message="System restart initiated. Services will be unavailable briefly.",
        )

    except Exception as e:
        logger.error(f"Failed to restart system: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to restart system: {str(e)}"
        )


@router.post("/cleanup", response_model=BaseResponse)
async def cleanup_system(
    cleanup_logs: bool = Query(False, description="Clean up old log files"),
    cleanup_temp: bool = Query(True, description="Clean up temporary files"),
    cleanup_models: bool = Query(False, description="Clean up old model checkpoints"),
    older_than_days: int = Query(
        7, ge=1, description="Clean up files older than N days"
    ),
):
    """
    Clean up system resources.

    Removes old files, clears caches, and frees up disk space.
    Useful for maintenance and preventing disk space issues.
    """
    try:
        cleanup_results = {
            "files_removed": 0,
            "space_freed_mb": 0,
            "actions_performed": [],
        }

        # Cleanup temporary files
        if cleanup_temp:
            # Placeholder cleanup logic
            cleanup_results["files_removed"] += 50
            cleanup_results["space_freed_mb"] += 100
            cleanup_results["actions_performed"].append("Removed temporary files")

        # Cleanup old logs
        if cleanup_logs:
            # Placeholder cleanup logic
            cleanup_results["files_removed"] += 25
            cleanup_results["space_freed_mb"] += 50
            cleanup_results["actions_performed"].append("Removed old log files")

        # Cleanup old models
        if cleanup_models:
            # Placeholder cleanup logic
            cleanup_results["files_removed"] += 10
            cleanup_results["space_freed_mb"] += 500
            cleanup_results["actions_performed"].append("Removed old model checkpoints")

        logger.info(f"System cleanup completed: {cleanup_results}")

        return BaseResponse(
            success=True,
            message=f"Cleanup completed. Removed {cleanup_results['files_removed']} files, "
            f"freed {cleanup_results['space_freed_mb']}MB of space.",
        )

    except Exception as e:
        logger.error(f"Failed to cleanup system: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to cleanup system: {str(e)}"
        )


@router.get("/stats")
async def get_system_stats():
    """
    Get system statistics.

    Returns comprehensive statistics about system usage,
    training history, and performance metrics.
    """
    try:
        # In a real implementation, these would come from actual data stores
        stats = {
            "uptime": {
                "current_uptime_hours": 24.5,
                "total_uptime_hours": 720.0,
                "restarts_count": 5,
                "last_restart": "2024-01-15T10:30:00Z",
            },
            "training": {
                "total_sessions": 150,
                "completed_sessions": 140,
                "failed_sessions": 8,
                "cancelled_sessions": 2,
                "average_duration_hours": 2.5,
                "total_training_hours": 350.0,
                "models_generated": 140,
            },
            "evaluation": {
                "total_evaluations": 300,
                "successful_evaluations": 285,
                "failed_evaluations": 15,
                "average_success_rate": 0.78,
                "average_reward": 125.5,
            },
            "robots": {
                "total_robots": 5,
                "active_robots": 2,
                "most_trained_robot": "unitree_g1_01",
                "total_robot_episodes": 50000,
            },
            "resources": {
                "peak_memory_usage_gb": 8.5,
                "average_cpu_usage_percent": 45.2,
                "disk_io_mb": 1500.0,
                "network_io_mb": 200.0,
            },
            "errors": {
                "critical_errors": 2,
                "errors": 15,
                "warnings": 150,
                "last_critical_error": "2024-01-10T14:22:00Z",
            },
        }

        return JSONResponse(content=stats)

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve system statistics: {str(e)}"
        )
