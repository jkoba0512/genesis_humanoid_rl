"""
Health check endpoints for Genesis Humanoid RL API.

Provides health monitoring and status endpoints for the API service.
"""

import logging
import time
import psutil
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..models import HealthCheckResponse, DetailedHealthCheck
from ...infrastructure.monitoring.genesis_monitor import check_genesis_status

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=HealthCheckResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns basic health status and system information.
    """
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        dependencies={
            "genesis": "available",
            "database": "available",
            "filesystem": "available",
        },
    )


@router.get("/live", response_model=HealthCheckResponse)
async def liveness_probe():
    """
    Kubernetes liveness probe endpoint.

    Returns 200 if the service is running, regardless of dependencies.
    """
    return HealthCheckResponse(status="alive", version="1.0.0")


@router.get("/ready", response_model=HealthCheckResponse)
async def readiness_probe():
    """
    Kubernetes readiness probe endpoint.

    Returns 200 only if the service is ready to accept traffic.
    """
    # Check critical dependencies
    dependencies = {}
    ready = True

    # Check Genesis
    try:
        genesis_status = check_genesis_status()
        if genesis_status["available"]:
            dependencies["genesis"] = "ready"
        else:
            dependencies["genesis"] = "not_ready"
            ready = False
    except Exception as e:
        logger.error(f"Genesis check failed: {e}")
        dependencies["genesis"] = "error"
        ready = False

    # Check database (simplified - assume available for now)
    dependencies["database"] = "ready"

    # Check filesystem
    try:
        # Simple filesystem check
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"test")
        dependencies["filesystem"] = "ready"
    except Exception as e:
        logger.error(f"Filesystem check failed: {e}")
        dependencies["filesystem"] = "error"
        ready = False

    status_code = 200 if ready else 503

    response = HealthCheckResponse(
        status="ready" if ready else "not_ready",
        version="1.0.0",
        dependencies=dependencies,
    )

    return JSONResponse(
        status_code=status_code, content=response.model_dump(mode="json")
    )


@router.get("/detailed", response_model=DetailedHealthCheck)
async def detailed_health_check(request: Request):
    """
    Detailed health check with component status and metrics.

    Provides comprehensive health information including system resources,
    component status, and recent errors.
    """
    startup_time = getattr(request.app.state, "startup_time", time.time())
    uptime = time.time() - startup_time

    # System resources
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        system_resources = {
            "cpu": {
                "count": psutil.cpu_count(),
                "percent": psutil.cpu_percent(interval=1),
                "load_avg": (
                    psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                ),
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": memory.percent,
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "percent_used": (disk.used / disk.total) * 100,
            },
        }
    except Exception as e:
        logger.error(f"Failed to get system resources: {e}")
        system_resources = {"error": str(e)}

    # Component health checks
    components = {}

    # Genesis component
    try:
        genesis_status = check_genesis_status()
        components["genesis"] = {
            "status": "healthy" if genesis_status["available"] else "unhealthy",
            "version": genesis_status.get("version", "unknown"),
            "details": genesis_status,
        }
    except Exception as e:
        components["genesis"] = {"status": "error", "error": str(e)}

    # Database component (simplified)
    components["database"] = {
        "status": "healthy",
        "type": "sqlite",
        "details": {"connection": "available"},
    }

    # Filesystem component
    try:
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            f.write(b"health_check_test")
        components["filesystem"] = {
            "status": "healthy",
            "details": {"read_write": "available"},
        }
    except Exception as e:
        components["filesystem"] = {"status": "error", "error": str(e)}

    # API component
    components["api"] = {
        "status": "healthy",
        "uptime_seconds": uptime,
        "config": getattr(request.app.state, "config", {}),
    }

    # Recent errors (placeholder - would integrate with logging system)
    recent_errors = []

    # Determine overall status
    unhealthy_components = [
        name
        for name, component in components.items()
        if component.get("status") != "healthy"
    ]

    if not unhealthy_components:
        overall_status = "healthy"
    elif len(unhealthy_components) < len(components) / 2:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    return DetailedHealthCheck(
        status=overall_status,
        version="1.0.0",
        uptime=uptime,
        dependencies={
            name: component.get("status", "unknown")
            for name, component in components.items()
        },
        components=components,
        system_resources=system_resources,
        recent_errors=recent_errors,
    )


@router.get("/metrics")
async def health_metrics():
    """
    Prometheus-style metrics endpoint.

    Returns metrics in Prometheus text format for monitoring systems.
    """
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Check Genesis
        genesis_status = check_genesis_status()
        genesis_available = 1 if genesis_status["available"] else 0

        # Format as Prometheus metrics
        metrics = f"""# HELP genesis_rl_api_up API service status
# TYPE genesis_rl_api_up gauge
genesis_rl_api_up 1

# HELP genesis_rl_api_info API information
# TYPE genesis_rl_api_info gauge
genesis_rl_api_info{{version="1.0.0"}} 1

# HELP genesis_rl_genesis_available Genesis availability
# TYPE genesis_rl_genesis_available gauge
genesis_rl_genesis_available {genesis_available}

# HELP genesis_rl_cpu_percent CPU usage percentage
# TYPE genesis_rl_cpu_percent gauge
genesis_rl_cpu_percent {cpu_percent}

# HELP genesis_rl_memory_percent Memory usage percentage
# TYPE genesis_rl_memory_percent gauge
genesis_rl_memory_percent {memory.percent}

# HELP genesis_rl_disk_percent Disk usage percentage
# TYPE genesis_rl_disk_percent gauge
genesis_rl_disk_percent {(disk.used / disk.total) * 100:.2f}

# HELP genesis_rl_memory_bytes Memory usage in bytes
# TYPE genesis_rl_memory_bytes gauge
genesis_rl_memory_bytes{{type="total"}} {memory.total}
genesis_rl_memory_bytes{{type="available"}} {memory.available}
genesis_rl_memory_bytes{{type="used"}} {memory.used}

# HELP genesis_rl_disk_bytes Disk usage in bytes
# TYPE genesis_rl_disk_bytes gauge
genesis_rl_disk_bytes{{type="total"}} {disk.total}
genesis_rl_disk_bytes{{type="free"}} {disk.free}
genesis_rl_disk_bytes{{type="used"}} {disk.used}
"""

        return JSONResponse(content=metrics, media_type="text/plain")

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return JSONResponse(
            status_code=500, content={"error": "Failed to generate metrics"}
        )
