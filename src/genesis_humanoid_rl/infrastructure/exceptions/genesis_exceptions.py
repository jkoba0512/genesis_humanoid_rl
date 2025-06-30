"""
Genesis-specific exceptions and error mapping.

Provides comprehensive error handling for Genesis physics engine integration,
including automatic error detection, classification, and recovery strategies.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class GenesisErrorType(Enum):
    """Classification of Genesis error types."""

    # Physics errors
    PHYSICS_INSTABILITY = "physics_instability"
    COLLISION_DETECTION = "collision_detection"
    CONSTRAINT_VIOLATION = "constraint_violation"
    SOLVER_FAILURE = "solver_failure"

    # Robot errors
    ROBOT_LOADING = "robot_loading"
    JOINT_LIMITS = "joint_limits"
    URDF_PARSING = "urdf_parsing"
    MESH_LOADING = "mesh_loading"

    # Scene errors
    SCENE_CREATION = "scene_creation"
    ENTITY_ADDITION = "entity_addition"
    MATERIAL_ASSIGNMENT = "material_assignment"
    CAMERA_SETUP = "camera_setup"

    # API version errors
    VERSION_MISMATCH = "version_mismatch"
    DEPRECATED_API = "deprecated_api"
    MISSING_FEATURE = "missing_feature"

    # Resource errors
    MEMORY_EXHAUSTION = "memory_exhaustion"
    GPU_ERROR = "gpu_error"
    FILE_NOT_FOUND = "file_not_found"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error diagnosis and recovery."""

    error_type: GenesisErrorType
    original_exception: Exception
    error_message: str
    stack_trace: str
    recovery_suggestions: list[str]
    severity: str  # 'low', 'medium', 'high', 'critical'
    retryable: bool
    component: str
    operation: str
    timestamp: float
    metadata: Dict[str, Any]


class GenesisError(Exception):
    """Base exception for all Genesis-related errors."""

    def __init__(
        self,
        message: str,
        error_type: GenesisErrorType = GenesisErrorType.UNKNOWN,
        original_exception: Optional[Exception] = None,
        recovery_suggestions: Optional[list[str]] = None,
        severity: str = "medium",
        retryable: bool = False,
        component: str = "genesis",
        operation: str = "unknown",
        **metadata,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.original_exception = original_exception
        self.recovery_suggestions = recovery_suggestions or []
        self.severity = severity
        self.retryable = retryable
        self.component = component
        self.operation = operation
        self.metadata = metadata

        # Create error context
        import time

        self.context = ErrorContext(
            error_type=error_type,
            original_exception=original_exception or self,
            error_message=message,
            stack_trace=traceback.format_exc(),
            recovery_suggestions=self.recovery_suggestions,
            severity=severity,
            retryable=retryable,
            component=component,
            operation=operation,
            timestamp=time.time(),
            metadata=metadata,
        )


class GenesisSimulationError(GenesisError):
    """Errors related to Genesis simulation execution."""

    def __init__(self, message: str, **kwargs):
        # Only set default error_type if not provided
        if "error_type" not in kwargs:
            kwargs["error_type"] = GenesisErrorType.PHYSICS_INSTABILITY
        if "component" not in kwargs:
            kwargs["component"] = "simulation"
        super().__init__(message, **kwargs)


class GenesisPhysicsError(GenesisError):
    """Errors related to Genesis physics calculations."""

    def __init__(self, message: str, **kwargs):
        # Only set defaults if not provided
        if "error_type" not in kwargs:
            kwargs["error_type"] = GenesisErrorType.SOLVER_FAILURE
        if "component" not in kwargs:
            kwargs["component"] = "physics"
        if "severity" not in kwargs:
            kwargs["severity"] = "high"
        super().__init__(message, **kwargs)


class GenesisRobotError(GenesisError):
    """Errors related to robot loading and configuration."""

    def __init__(self, message: str, **kwargs):
        if "error_type" not in kwargs:
            kwargs["error_type"] = GenesisErrorType.ROBOT_LOADING
        if "component" not in kwargs:
            kwargs["component"] = "robot"
        super().__init__(message, **kwargs)


class GenesisSceneError(GenesisError):
    """Errors related to scene creation and management."""

    def __init__(self, message: str, **kwargs):
        if "error_type" not in kwargs:
            kwargs["error_type"] = GenesisErrorType.SCENE_CREATION
        if "component" not in kwargs:
            kwargs["component"] = "scene"
        super().__init__(message, **kwargs)


class GenesisVersionError(GenesisError):
    """Errors related to Genesis API version compatibility."""

    def __init__(self, message: str, **kwargs):
        if "error_type" not in kwargs:
            kwargs["error_type"] = GenesisErrorType.VERSION_MISMATCH
        if "component" not in kwargs:
            kwargs["component"] = "api"
        if "severity" not in kwargs:
            kwargs["severity"] = "critical"
        super().__init__(message, **kwargs)


class GenesisExceptionMapper:
    """Maps Genesis exceptions to domain-specific exceptions with recovery strategies."""

    # Error pattern matching for Genesis exceptions
    ERROR_PATTERNS = {
        # Physics instability patterns
        GenesisErrorType.PHYSICS_INSTABILITY: [
            "physics instability",
            "simulation unstable",
            "solver diverged",
            "nan values detected",
            "infinite force",
            "exploding simulation",
        ],
        # Collision detection patterns
        GenesisErrorType.COLLISION_DETECTION: [
            "collision detection failed",
            "overlapping bodies",
            "penetration depth",
            "contact normal",
        ],
        # Robot loading patterns
        GenesisErrorType.ROBOT_LOADING: [
            "failed to load robot",
            "urdf parsing error",
            "mesh not found",
            "invalid joint",
            "missing link",
        ],
        # Scene errors
        GenesisErrorType.SCENE_CREATION: [
            "scene creation failed",
            "failed to add entity",
            "material not found",
            "camera setup failed",
        ],
        # Version mismatch patterns
        GenesisErrorType.VERSION_MISMATCH: [
            "api not available",
            "method not found",
            "deprecated function",
            "version mismatch",
        ],
        # Memory/Resource patterns
        GenesisErrorType.MEMORY_EXHAUSTION: [
            "out of memory",
            "cuda out of memory",
            "allocation failed",
            "memory exhausted",
        ],
        # GPU errors
        GenesisErrorType.GPU_ERROR: [
            "cuda error",
            "gpu device error",
            "curand",
            "cudnn error",
        ],
    }

    # Recovery strategies for each error type
    RECOVERY_STRATEGIES = {
        GenesisErrorType.PHYSICS_INSTABILITY: [
            "Reduce simulation timestep",
            "Increase solver iterations",
            "Check for extreme forces or positions",
            "Verify joint limits and constraints",
            "Consider simulation parameter tuning",
        ],
        GenesisErrorType.COLLISION_DETECTION: [
            "Check robot initial position",
            "Verify collision meshes",
            "Adjust ground clearance",
            "Review collision margins",
        ],
        GenesisErrorType.ROBOT_LOADING: [
            "Verify URDF file path and format",
            "Check mesh file availability",
            "Validate joint definitions",
            "Ensure asset files are accessible",
        ],
        GenesisErrorType.SCENE_CREATION: [
            "Check Genesis installation",
            "Verify scene parameters",
            "Review entity configurations",
            "Ensure proper initialization order",
        ],
        GenesisErrorType.VERSION_MISMATCH: [
            "Update Genesis to latest version",
            "Check API documentation for changes",
            "Use compatibility layer if available",
            "Review migration guide",
        ],
        GenesisErrorType.MEMORY_EXHAUSTION: [
            "Reduce batch size or environment count",
            "Clear GPU memory cache",
            "Use mixed precision training",
            "Monitor memory usage patterns",
        ],
        GenesisErrorType.GPU_ERROR: [
            "Check CUDA installation",
            "Verify GPU driver compatibility",
            "Reset GPU state",
            "Use CPU fallback if available",
        ],
    }

    @classmethod
    def classify_error(cls, exception: Exception) -> GenesisErrorType:
        """Classify an exception based on error message patterns."""

        error_message = str(exception).lower()

        for error_type, patterns in cls.ERROR_PATTERNS.items():
            for pattern in patterns:
                if pattern in error_message:
                    return error_type

        return GenesisErrorType.UNKNOWN

    @classmethod
    def map_exception(
        cls,
        exception: Exception,
        operation: str = "unknown",
        component: str = "genesis",
    ) -> GenesisError:
        """Map a Genesis exception to a domain-specific exception with context."""

        error_type = cls.classify_error(exception)
        recovery_suggestions = cls.RECOVERY_STRATEGIES.get(error_type, [])

        # Determine severity and retryability
        severity = "medium"
        retryable = False

        if error_type in [
            GenesisErrorType.PHYSICS_INSTABILITY,
            GenesisErrorType.SOLVER_FAILURE,
        ]:
            severity = "high"
            retryable = True
        elif error_type in [
            GenesisErrorType.MEMORY_EXHAUSTION,
            GenesisErrorType.GPU_ERROR,
        ]:
            severity = "high"
            retryable = True
        elif error_type == GenesisErrorType.VERSION_MISMATCH:
            severity = "critical"
            retryable = False

        # Create appropriate exception type
        if error_type in [
            GenesisErrorType.PHYSICS_INSTABILITY,
            GenesisErrorType.SOLVER_FAILURE,
        ]:
            exception_class = GenesisPhysicsError
        elif error_type in [
            GenesisErrorType.ROBOT_LOADING,
            GenesisErrorType.URDF_PARSING,
        ]:
            exception_class = GenesisRobotError
        elif error_type in [
            GenesisErrorType.SCENE_CREATION,
            GenesisErrorType.ENTITY_ADDITION,
        ]:
            exception_class = GenesisSceneError
        elif error_type == GenesisErrorType.VERSION_MISMATCH:
            exception_class = GenesisVersionError
        else:
            exception_class = GenesisSimulationError

        return exception_class(
            message=f"Genesis {error_type.value}: {str(exception)}",
            error_type=error_type,
            original_exception=exception,
            recovery_suggestions=recovery_suggestions,
            severity=severity,
            retryable=retryable,
            component=component,
            operation=operation,
            genesis_version=cls._get_genesis_version(),
            python_version=cls._get_python_version(),
        )

    @staticmethod
    def _get_genesis_version() -> str:
        """Get Genesis version for error context."""
        try:
            import genesis as gs

            return getattr(gs, "__version__", "unknown")
        except ImportError:
            return "not_installed"

    @staticmethod
    def _get_python_version() -> str:
        """Get Python version for error context."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def map_genesis_exception(
    exception: Exception, operation: str = "unknown", component: str = "genesis"
) -> GenesisError:
    """
    Convenience function to map Genesis exceptions.

    Args:
        exception: The original exception
        operation: The operation being performed when error occurred
        component: The component where error occurred

    Returns:
        Mapped GenesisError with recovery context
    """
    return GenesisExceptionMapper.map_exception(exception, operation, component)


def create_error_context(
    error: GenesisError, additional_metadata: Optional[Dict[str, Any]] = None
) -> ErrorContext:
    """
    Create enhanced error context for logging and monitoring.

    Args:
        error: The GenesisError instance
        additional_metadata: Additional context metadata

    Returns:
        Enhanced ErrorContext with diagnostic information
    """
    context = error.context

    if additional_metadata:
        context.metadata.update(additional_metadata)

    # Add system information
    context.metadata.update(
        {
            "genesis_version": GenesisExceptionMapper._get_genesis_version(),
            "python_version": GenesisExceptionMapper._get_python_version(),
            "platform": _get_platform_info(),
        }
    )

    return context


def _get_platform_info() -> Dict[str, str]:
    """Get platform information for error context."""
    import platform

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
    }
