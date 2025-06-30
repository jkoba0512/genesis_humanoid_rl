"""
Infrastructure-level exceptions for external services and resources.

Provides error handling for database connections, external APIs,
configuration services, and system resource management.
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class InfrastructureErrorType(Enum):
    """Classification of infrastructure error types."""

    # Database errors
    DATABASE_CONNECTION = "database_connection"
    DATABASE_QUERY = "database_query"
    DATABASE_TRANSACTION = "database_transaction"
    DATABASE_MIGRATION = "database_migration"

    # External service errors
    API_UNAVAILABLE = "api_unavailable"
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"

    # Resource errors
    MEMORY_LIMIT = "memory_limit"
    DISK_SPACE = "disk_space"
    CPU_LIMIT = "cpu_limit"
    GPU_UNAVAILABLE = "gpu_unavailable"

    # Configuration errors
    CONFIG_MISSING = "config_missing"
    CONFIG_INVALID = "config_invalid"
    CONFIG_PERMISSION = "config_permission"

    # Performance errors
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TIMEOUT = "timeout"
    DEADLOCK = "deadlock"

    # Network errors
    NETWORK_UNAVAILABLE = "network_unavailable"
    DNS_RESOLUTION = "dns_resolution"
    CONNECTION_REFUSED = "connection_refused"


@dataclass
class InfrastructureErrorContext:
    """Context information for infrastructure errors."""

    error_type: InfrastructureErrorType
    component: str
    operation: str
    resource_usage: Dict[str, float]
    retry_count: int
    max_retries: int
    backoff_delay: float
    metadata: Dict[str, Any]


class InfrastructureError(Exception):
    """Base exception for infrastructure-related errors."""

    def __init__(
        self,
        message: str,
        error_type: InfrastructureErrorType,
        component: str = "unknown",
        operation: str = "unknown",
        retryable: bool = True,
        max_retries: int = 3,
        backoff_delay: float = 1.0,
        **metadata,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.component = component
        self.operation = operation
        self.retryable = retryable
        self.max_retries = max_retries
        self.backoff_delay = backoff_delay
        self.metadata = metadata

        self.context = InfrastructureErrorContext(
            error_type=error_type,
            component=component,
            operation=operation,
            resource_usage=self._get_resource_usage(),
            retry_count=0,
            max_retries=max_retries,
            backoff_delay=backoff_delay,
            metadata=metadata,
        )

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage."""
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            }
        except ImportError:
            return {}


class ExternalServiceError(InfrastructureError):
    """Error in external service communication."""

    def __init__(self, message: str, service_name: str, **kwargs):
        super().__init__(
            message,
            error_type=InfrastructureErrorType.API_UNAVAILABLE,
            component=f"external_service.{service_name}",
            **kwargs,
        )
        self.service_name = service_name


class ResourceError(InfrastructureError):
    """Error related to system resource constraints."""

    def __init__(self, message: str, resource_type: str, **kwargs):
        super().__init__(
            message,
            error_type=InfrastructureErrorType.MEMORY_LIMIT,
            component=f"resource.{resource_type}",
            **kwargs,
        )
        self.resource_type = resource_type


class ConfigurationError(InfrastructureError):
    """Error in configuration management."""

    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(
            message,
            error_type=InfrastructureErrorType.CONFIG_INVALID,
            component="configuration",
            retryable=False,  # Config errors usually aren't retryable
            **kwargs,
        )
        self.config_key = config_key


class PerformanceError(InfrastructureError):
    """Error indicating performance degradation."""

    def __init__(
        self,
        message: str,
        metric_name: str,
        threshold: float,
        actual_value: float,
        **kwargs,
    ):
        super().__init__(
            message,
            error_type=InfrastructureErrorType.PERFORMANCE_DEGRADATION,
            component="performance",
            **kwargs,
        )
        self.metric_name = metric_name
        self.threshold = threshold
        self.actual_value = actual_value


class DatabaseError(InfrastructureError):
    """Database-specific error with connection and query context."""

    def __init__(
        self, message: str, database_name: str = None, query: str = None, **kwargs
    ):
        super().__init__(
            message,
            error_type=InfrastructureErrorType.DATABASE_CONNECTION,
            component="database",
            **kwargs,
        )
        self.database_name = database_name
        self.query = query


class InfrastructureErrorHandler:
    """Centralized handler for infrastructure errors with retry logic."""

    def __init__(self):
        self.error_counts = {}
        self.circuit_breakers = {}

    def handle_error(
        self,
        error: InfrastructureError,
        operation_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Handle infrastructure error with appropriate recovery strategy.

        Args:
            error: The infrastructure error that occurred
            operation_context: Additional context about the operation

        Returns:
            Recovery action plan
        """
        error_key = f"{error.component}.{error.operation}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # Log error with context
        self._log_infrastructure_error(error, operation_context)

        # Check circuit breaker
        if self._should_circuit_break(error_key):
            return self._create_circuit_breaker_response(error)

        # Determine recovery strategy based on error type
        if error.error_type in [
            InfrastructureErrorType.DATABASE_CONNECTION,
            InfrastructureErrorType.API_UNAVAILABLE,
            InfrastructureErrorType.NETWORK_UNAVAILABLE,
        ]:
            return self._handle_connection_error(error)

        elif error.error_type in [
            InfrastructureErrorType.MEMORY_LIMIT,
            InfrastructureErrorType.CPU_LIMIT,
            InfrastructureErrorType.DISK_SPACE,
        ]:
            return self._handle_resource_error(error)

        elif error.error_type in [
            InfrastructureErrorType.CONFIG_MISSING,
            InfrastructureErrorType.CONFIG_INVALID,
        ]:
            return self._handle_configuration_error(error)

        elif error.error_type == InfrastructureErrorType.PERFORMANCE_DEGRADATION:
            return self._handle_performance_error(error)

        else:
            return self._handle_generic_error(error)

    def _handle_connection_error(self, error: InfrastructureError) -> Dict[str, Any]:
        """Handle connection-related errors with exponential backoff."""

        return {
            "action": "retry_with_backoff",
            "strategy": {
                "retry_count": error.context.retry_count + 1,
                "max_retries": error.max_retries,
                "backoff_delay": error.backoff_delay * (2**error.context.retry_count),
                "circuit_breaker_threshold": 5,
            },
            "recovery_actions": [
                "Check network connectivity",
                "Verify service endpoints",
                "Test authentication credentials",
                "Review firewall rules",
            ],
        }

    def _handle_resource_error(self, error: InfrastructureError) -> Dict[str, Any]:
        """Handle resource constraint errors."""

        return {
            "action": "resource_optimization",
            "strategy": {
                "reduce_batch_size": True,
                "enable_memory_cleanup": True,
                "use_disk_caching": True,
                "scale_horizontally": True,
            },
            "recovery_actions": [
                "Clear memory caches",
                "Reduce concurrent operations",
                "Optimize data structures",
                "Consider resource scaling",
            ],
        }

    def _handle_configuration_error(self, error: InfrastructureError) -> Dict[str, Any]:
        """Handle configuration errors."""

        return {
            "action": "configuration_repair",
            "strategy": {
                "validate_config": True,
                "use_defaults": True,
                "reload_config": True,
                "check_permissions": True,
            },
            "recovery_actions": [
                "Validate configuration syntax",
                "Check file permissions",
                "Verify environment variables",
                "Use fallback configuration",
            ],
        }

    def _handle_performance_error(self, error: InfrastructureError) -> Dict[str, Any]:
        """Handle performance degradation."""

        return {
            "action": "performance_optimization",
            "strategy": {
                "enable_profiling": True,
                "optimize_queries": True,
                "increase_cache_size": True,
                "reduce_complexity": True,
            },
            "recovery_actions": [
                "Profile performance bottlenecks",
                "Optimize critical paths",
                "Increase resource allocation",
                "Review algorithm complexity",
            ],
        }

    def _handle_generic_error(self, error: InfrastructureError) -> Dict[str, Any]:
        """Handle generic infrastructure errors."""

        return {
            "action": "generic_recovery",
            "strategy": {
                "retry_with_delay": True,
                "enable_fallback": True,
                "increase_timeout": True,
                "enable_debug_mode": True,
            },
            "recovery_actions": [
                "Retry operation with increased timeout",
                "Enable fallback mechanisms",
                "Collect additional diagnostics",
                "Contact system administrator",
            ],
        }

    def _should_circuit_break(self, error_key: str) -> bool:
        """Check if circuit breaker should trigger."""
        error_count = self.error_counts.get(error_key, 0)
        return error_count > 10  # Circuit break after 10 consecutive errors

    def _create_circuit_breaker_response(
        self, error: InfrastructureError
    ) -> Dict[str, Any]:
        """Create circuit breaker response."""

        return {
            "action": "circuit_breaker_open",
            "strategy": {
                "disable_component": True,
                "use_fallback_only": True,
                "cooldown_period": 300,  # 5 minutes
                "alert_administrators": True,
            },
            "recovery_actions": [
                "Component temporarily disabled",
                "Using fallback mechanisms",
                "Manual intervention required",
                "Check system health",
            ],
        }

    def _log_infrastructure_error(
        self, error: InfrastructureError, operation_context: Optional[Dict[str, Any]]
    ) -> None:
        """Log infrastructure error with full context."""

        error_info = {
            "error_type": error.error_type.value,
            "component": error.component,
            "operation": error.operation,
            "message": str(error),
            "retryable": error.retryable,
            "resource_usage": error.context.resource_usage,
        }

        if operation_context:
            error_info["operation_context"] = operation_context

        logger.error(f"Infrastructure error: {error_info}")

    def reset_error_counts(self, component: str = None) -> None:
        """Reset error counts for component or all components."""
        if component:
            keys_to_remove = [
                k for k in self.error_counts.keys() if k.startswith(component)
            ]
            for key in keys_to_remove:
                del self.error_counts[key]
        else:
            self.error_counts.clear()

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_by_component": self.error_counts.copy(),
            "circuit_breakers": list(self.circuit_breakers.keys()),
        }
