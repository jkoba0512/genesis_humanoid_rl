"""
API endpoints package for Genesis Humanoid RL.

Provides REST API endpoints organized by functionality.
"""

from .health import router as health_router
from .system import router as system_router
from .training import router as training_router
from .evaluation import router as evaluation_router
from .robots import router as robots_router
from .monitoring import router as monitoring_router

__all__ = [
    "health_router",
    "system_router",
    "training_router",
    "evaluation_router",
    "robots_router",
    "monitoring_router",
]
