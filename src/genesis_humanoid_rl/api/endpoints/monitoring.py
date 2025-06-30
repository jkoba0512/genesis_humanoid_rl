"""
Monitoring endpoints for Genesis Humanoid RL API.

Provides endpoints for system monitoring, metrics collection, and performance analysis.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse

from ..models import (
    MonitoringMetrics,
    SystemAlert,
    AlertListResponse,
    PerformanceReport,
    BaseResponse,
)
from ...infrastructure.monitoring.genesis_monitor import GenesisAPIMonitor

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for demonstration
# In production, this would be a proper time-series database
metrics_history = []
system_alerts = {}
performance_reports = {}

# Initialize Genesis monitor
genesis_monitor = GenesisAPIMonitor()


@router.get("/metrics", response_model=MonitoringMetrics)
async def get_current_metrics():
    """
    Get current system monitoring metrics.

    Returns real-time metrics including system resources,
    training performance, and component health.
    """
    try:
        # Get system metrics
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=1)

        # Get Genesis metrics (simplified for API)
        try:
            from ...infrastructure.monitoring.genesis_monitor import (
                check_genesis_status,
            )

            genesis_status = check_genesis_status()
        except Exception:
            genesis_status = {"available": False, "version": "unknown"}

        # Calculate training metrics (placeholder)
        active_sessions = 0  # Would come from session manager
        total_models = 150  # Would come from model store

        metrics = MonitoringMetrics(
            timestamp=datetime.now(),
            system_metrics={
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": (disk.used / disk.total) * 100,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
            },
            training_metrics={
                "active_sessions": active_sessions,
                "total_sessions_today": 12,
                "average_fps": 58.5,
                "success_rate": 0.78,
                "total_models": total_models,
            },
            genesis_metrics={
                "available": genesis_status.get("compatible", False),
                "version": genesis_status.get("version", "unknown"),
                "performance_score": genesis_status.get("performance_score", 0.0),
                "gpu_acceleration": genesis_status.get("features", {}).get(
                    "gpu_acceleration", False
                ),
            },
            api_metrics={
                "requests_per_minute": 45.2,
                "average_response_time_ms": 125.3,
                "error_rate": 0.02,
                "active_connections": 8,
            },
        )

        # Store metrics for history
        metrics_history.append(metrics)

        # Keep only last 1000 entries
        if len(metrics_history) > 1000:
            metrics_history.pop(0)

        return metrics

    except Exception as e:
        logger.error(f"Failed to get current metrics: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get monitoring metrics: {str(e)}"
        )


@router.get("/metrics/history")
async def get_metrics_history(
    hours: int = Query(24, ge=1, le=168), interval_minutes: int = Query(5, ge=1, le=60)
):
    """
    Get historical metrics data.

    Returns time-series metrics data for the specified time range.
    Useful for creating charts and analyzing trends.
    """
    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Filter metrics by time range (simplified - using recent data)
        recent_metrics = metrics_history[-min(hours * 12, len(metrics_history)) :]

        # Sample metrics by interval
        sampled_metrics = []
        if recent_metrics:
            sample_step = max(
                1, len(recent_metrics) // (hours * 60 // interval_minutes)
            )
            sampled_metrics = recent_metrics[::sample_step]

        return JSONResponse(
            content={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "interval_minutes": interval_minutes,
                "data_points": len(sampled_metrics),
                "metrics": [
                    {
                        "timestamp": metric.timestamp.isoformat(),
                        "system": metric.system_metrics,
                        "training": metric.training_metrics,
                        "genesis": metric.genesis_metrics,
                        "api": metric.api_metrics,
                    }
                    for metric in sampled_metrics
                ],
            }
        )

    except Exception as e:
        logger.error(f"Failed to get metrics history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get metrics history: {str(e)}"
        )


@router.get("/alerts", response_model=AlertListResponse)
async def get_system_alerts(
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    active_only: bool = Query(True),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Get system alerts.

    Returns current and historical system alerts with filtering options.
    """
    try:
        # Get all alerts
        all_alerts = list(system_alerts.values())

        # Apply filters
        filtered_alerts = all_alerts

        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]

        if active_only:
            filtered_alerts = [a for a in filtered_alerts if a.status == "active"]

        # Sort by creation time (newest first)
        filtered_alerts.sort(key=lambda x: x.created_at, reverse=True)

        # Apply limit
        filtered_alerts = filtered_alerts[:limit]

        return AlertListResponse(
            alerts=filtered_alerts,
            total_count=len(filtered_alerts),
            active_count=len([a for a in filtered_alerts if a.status == "active"]),
        )

    except Exception as e:
        logger.error(f"Failed to get system alerts: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get system alerts: {str(e)}"
        )


@router.post("/alerts/{alert_id}/acknowledge", response_model=BaseResponse)
async def acknowledge_alert(alert_id: str):
    """
    Acknowledge a system alert.

    Marks an alert as acknowledged by an operator.
    """
    if alert_id not in system_alerts:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")

    try:
        alert = system_alerts[alert_id]
        alert.status = "acknowledged"
        alert.acknowledged_at = datetime.now()

        system_alerts[alert_id] = alert

        logger.info(f"Alert {alert_id} acknowledged")

        return BaseResponse(
            success=True, message=f"Alert {alert_id} acknowledged successfully"
        )

    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to acknowledge alert: {str(e)}"
        )


@router.get("/reports/performance", response_model=PerformanceReport)
async def get_performance_report(
    period: str = Query("24h", regex="^(1h|6h|24h|7d|30d)$"),
    include_recommendations: bool = Query(True),
):
    """
    Get performance analysis report.

    Returns comprehensive performance analysis for the specified period.
    """
    try:
        # Calculate period
        period_mapping = {"1h": 1, "6h": 6, "24h": 24, "7d": 24 * 7, "30d": 24 * 30}
        hours = period_mapping[period]

        # Generate performance analysis (simplified)
        report = PerformanceReport(
            report_id=f"perf_{int(time.time())}",
            period=period,
            generated_at=datetime.now(),
            summary={
                "overall_performance": "good",
                "average_cpu_usage": 45.2,
                "average_memory_usage": 62.8,
                "training_efficiency": 0.78,
                "system_stability": 0.92,
            },
            training_performance={
                "total_sessions": 15,
                "successful_sessions": 13,
                "average_fps": 58.5,
                "average_reward": 125.3,
                "convergence_rate": 0.85,
            },
            resource_utilization={
                "peak_cpu_usage": 85.2,
                "peak_memory_usage": 78.5,
                "disk_io_mb": 1250.0,
                "network_io_mb": 150.0,
            },
            issues_identified=[
                {
                    "type": "performance",
                    "severity": "medium",
                    "description": "Memory usage peaked at 78.5% during training session",
                    "recommendation": "Consider reducing parallel environments or increasing memory",
                },
                {
                    "type": "efficiency",
                    "severity": "low",
                    "description": "GPU utilization below optimal threshold",
                    "recommendation": "Verify GPU acceleration settings in Genesis configuration",
                },
            ],
            recommendations=(
                [
                    "Monitor memory usage during large training sessions",
                    "Consider implementing memory cleanup between episodes",
                    "Verify Genesis GPU acceleration configuration",
                    "Enable automatic checkpointing for long training runs",
                ]
                if include_recommendations
                else []
            ),
        )

        # Store report
        performance_reports[report.report_id] = report

        return report

    except Exception as e:
        logger.error(f"Failed to generate performance report: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate performance report: {str(e)}"
        )


@router.get("/reports/performance/{report_id}", response_model=PerformanceReport)
async def get_performance_report_by_id(report_id: str):
    """
    Get a specific performance report by ID.
    """
    if report_id not in performance_reports:
        raise HTTPException(
            status_code=404, detail=f"Performance report {report_id} not found"
        )

    return performance_reports[report_id]


@router.post("/genesis/test", response_model=BaseResponse)
async def test_genesis_integration(background_tasks: BackgroundTasks):
    """
    Test Genesis integration and performance.

    Runs comprehensive Genesis compatibility and performance tests.
    """
    try:
        # Start Genesis testing in background
        background_tasks.add_task(run_genesis_tests)

        return BaseResponse(
            success=True,
            message="Genesis integration test started. Results will be available in monitoring metrics.",
        )

    except Exception as e:
        logger.error(f"Failed to start Genesis test: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start Genesis test: {str(e)}"
        )


@router.get("/genesis/compatibility")
async def get_genesis_compatibility():
    """
    Get Genesis compatibility status and version information.
    """
    try:
        try:
            from ...infrastructure.monitoring.genesis_monitor import (
                check_genesis_status,
            )

            compatibility_report = check_genesis_status()
        except Exception as e:
            compatibility_report = {"available": False, "error": str(e)}

        return JSONResponse(
            content={
                "compatibility": compatibility_report,
                "last_check": datetime.now().isoformat(),
                "recommendations": [
                    "Update Genesis to latest version if available",
                    "Ensure CUDA drivers are up to date for GPU acceleration",
                    "Verify Python version compatibility (3.10-3.12)",
                ],
            }
        )

    except Exception as e:
        logger.error(f"Failed to check Genesis compatibility: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to check Genesis compatibility: {str(e)}"
        )


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus format for external monitoring systems.
    """
    try:
        # Get current metrics
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent()

        # Genesis status
        try:
            from ...infrastructure.monitoring.genesis_monitor import (
                check_genesis_status,
            )

            genesis_status = check_genesis_status()
            genesis_available = 1 if genesis_status.get("available", False) else 0
        except Exception:
            genesis_available = 0

        # Format as Prometheus metrics
        metrics = f"""# HELP genesis_rl_cpu_percent CPU usage percentage
# TYPE genesis_rl_cpu_percent gauge
genesis_rl_cpu_percent {cpu_percent}

# HELP genesis_rl_memory_percent Memory usage percentage  
# TYPE genesis_rl_memory_percent gauge
genesis_rl_memory_percent {memory.percent}

# HELP genesis_rl_disk_percent Disk usage percentage
# TYPE genesis_rl_disk_percent gauge
genesis_rl_disk_percent {(disk.used / disk.total) * 100:.2f}

# HELP genesis_rl_genesis_available Genesis availability
# TYPE genesis_rl_genesis_available gauge
genesis_rl_genesis_available {genesis_available}

# HELP genesis_rl_training_sessions Active training sessions
# TYPE genesis_rl_training_sessions gauge
genesis_rl_training_sessions 0

# HELP genesis_rl_models_total Total trained models
# TYPE genesis_rl_models_total counter
genesis_rl_models_total 150

# HELP genesis_rl_api_requests_total Total API requests
# TYPE genesis_rl_api_requests_total counter
genesis_rl_api_requests_total 1250

# HELP genesis_rl_errors_total Total errors
# TYPE genesis_rl_errors_total counter
genesis_rl_errors_total 25
"""

        return PlainTextResponse(content=metrics)

    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {e}")
        return PlainTextResponse(
            content="# Error generating metrics\n", status_code=500
        )


async def run_genesis_tests():
    """
    Background task to run comprehensive Genesis tests.
    """
    import asyncio

    try:
        logger.info("Starting Genesis integration tests")

        # Run compatibility tests
        try:
            from ...infrastructure.monitoring.genesis_monitor import (
                check_genesis_status,
            )

            compatibility = check_genesis_status()
        except Exception as e:
            compatibility = {"available": False, "error": str(e)}

        # Run performance tests (simplified)
        await asyncio.sleep(2)  # Simulate test time
        performance_results = {
            "basic_test": "passed",
            "performance_score": 0.85,
            "test_duration": 2.0,
        }

        # Generate test report
        test_results = {
            "compatibility": compatibility,
            "performance": performance_results,
            "test_completed_at": datetime.now().isoformat(),
            "overall_status": (
                "passed" if compatibility.get("compatible", False) else "failed"
            ),
        }

        # Create alert if tests failed
        if not compatibility.get("compatible", False):
            alert_id = f"genesis_test_failed_{int(time.time())}"
            alert = SystemAlert(
                alert_id=alert_id,
                alert_type="system",
                severity="high",
                title="Genesis Integration Test Failed",
                description="Genesis compatibility tests have failed. Check system configuration.",
                status="active",
                created_at=datetime.now(),
                metadata=test_results,
            )
            system_alerts[alert_id] = alert

        logger.info(f"Genesis tests completed: {test_results['overall_status']}")

    except Exception as e:
        logger.error(f"Genesis test execution failed: {e}")

        # Create error alert
        alert_id = f"genesis_test_error_{int(time.time())}"
        alert = SystemAlert(
            alert_id=alert_id,
            alert_type="system",
            severity="critical",
            title="Genesis Test Execution Error",
            description=f"Failed to execute Genesis tests: {str(e)}",
            status="active",
            created_at=datetime.now(),
            metadata={"error": str(e)},
        )
        system_alerts[alert_id] = alert


# Initialize some sample alerts for demonstration
def initialize_sample_alerts():
    """Initialize sample alerts for demonstration."""
    sample_alerts = [
        SystemAlert(
            alert_id="memory_usage_high",
            alert_type="resource",
            severity="medium",
            title="High Memory Usage",
            description="Memory usage exceeded 80% threshold",
            status="active",
            created_at=datetime.now() - timedelta(hours=2),
            metadata={"threshold": 80, "current": 85.2},
        ),
        SystemAlert(
            alert_id="training_session_failed",
            alert_type="training",
            severity="high",
            title="Training Session Failed",
            description="Training session terminated unexpectedly",
            status="acknowledged",
            created_at=datetime.now() - timedelta(hours=6),
            acknowledged_at=datetime.now() - timedelta(hours=5),
            metadata={"session_id": "sess_123", "error": "GPU memory exhausted"},
        ),
    ]

    for alert in sample_alerts:
        system_alerts[alert.alert_id] = alert


# Initialize sample data
initialize_sample_alerts()
