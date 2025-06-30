"""
Infrastructure exceptions module.

Provides comprehensive error handling and exception mapping for external systems.
"""

from .genesis_exceptions import *
from .simulation_exceptions import *
from .infrastructure_exceptions import *

__all__ = [
    # Genesis-specific exceptions
    "GenesisError",
    "GenesisSimulationError",
    "GenesisPhysicsError",
    "GenesisRobotError",
    "GenesisSceneError",
    "GenesisVersionError",
    # Simulation exceptions
    "SimulationError",
    "PhysicsInstabilityError",
    "RobotControlError",
    "EnvironmentError",
    "TrainingError",
    # Infrastructure exceptions
    "InfrastructureError",
    "ExternalServiceError",
    "ResourceError",
    "ConfigurationError",
    "PerformanceError",
    # Exception mapping utilities
    "map_genesis_exception",
    "handle_simulation_error",
    "create_error_context",
]
