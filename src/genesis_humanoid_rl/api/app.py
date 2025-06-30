"""
FastAPI application factory for Genesis Humanoid RL API.

Creates and configures the FastAPI application with all endpoints,
middleware, and configuration for the humanoid robotics training API.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from starlette.middleware.base import BaseHTTPMiddleware

from .endpoints import (
    training_router,
    evaluation_router,
    robots_router,
    monitoring_router,
    system_router,
    health_router,
)
from .models import ErrorResponse
from ..infrastructure.monitoring.genesis_monitor import check_genesis_status

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url}")

        try:
            response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response
            logger.info(
                f"Response: {request.method} {request.url} "
                f"- Status: {response.status_code} "
                f"- Time: {process_time:.3f}s"
            )

            # Add processing time to response headers
            response.headers["X-Process-Time"] = str(process_time)

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url} "
                f"- Error: {str(e)} - Time: {process_time:.3f}s"
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting middleware."""

    def __init__(self, app, calls_per_minute: int = 1000):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.client_calls: Dict[str, list] = {}

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host
        current_time = time.time()

        # Clean old entries
        if client_ip in self.client_calls:
            self.client_calls[client_ip] = [
                call_time
                for call_time in self.client_calls[client_ip]
                if current_time - call_time < 60  # Keep last minute only
            ]
        else:
            self.client_calls[client_ip] = []

        # Check rate limit
        if len(self.client_calls[client_ip]) >= self.calls_per_minute:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Too many requests per minute.",
            )

        # Record this call
        self.client_calls[client_ip].append(current_time)

        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    logger.info("Starting Genesis Humanoid RL API...")

    # Check Genesis status on startup
    genesis_status = check_genesis_status()
    if not genesis_status["available"]:
        logger.warning(f"Genesis not available: {genesis_status['status']}")
    else:
        logger.info(f"Genesis available: version {genesis_status['version']}")

    # Store startup time
    app.state.startup_time = time.time()

    # Initialize any background tasks here
    logger.info("API startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Genesis Humanoid RL API...")

    # Cleanup background tasks here
    logger.info("API shutdown complete")


def create_app(
    title: str = "Genesis Humanoid RL API",
    description: str = "REST API for Genesis-based humanoid robotics reinforcement learning",
    version: str = "1.0.0",
    debug: bool = False,
    enable_rate_limiting: bool = True,
    rate_limit_per_minute: int = 1000,
    cors_origins: Optional[list] = None,
) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        title: API title
        description: API description
        version: API version
        debug: Enable debug mode
        enable_rate_limiting: Enable rate limiting middleware
        rate_limit_per_minute: Rate limit per client per minute
        cors_origins: Allowed CORS origins

    Returns:
        Configured FastAPI application
    """

    # Create FastAPI app with lifespan
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        debug=debug,
        lifespan=lifespan,
        docs_url="/docs" if debug else None,
        redoc_url="/redoc" if debug else None,
        openapi_url="/openapi.json" if debug else None,
    )

    # Configure CORS
    if cors_origins is None:
        cors_origins = (
            [
                "http://localhost:3000",  # React dev server
                "http://localhost:8080",  # Vue dev server
                "http://localhost:5173",  # Vite dev server
            ]
            if debug
            else []
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Add rate limiting if enabled
    if enable_rate_limiting:
        app.add_middleware(RateLimitMiddleware, calls_per_minute=rate_limit_per_minute)

    # Add logging middleware
    app.add_middleware(LoggingMiddleware)

    # Include routers
    app.include_router(health_router, prefix="/health", tags=["Health"])
    app.include_router(system_router, prefix="/system", tags=["System"])
    app.include_router(training_router, prefix="/training", tags=["Training"])
    app.include_router(evaluation_router, prefix="/evaluation", tags=["Evaluation"])
    app.include_router(robots_router, prefix="/robots", tags=["Robots"])
    app.include_router(monitoring_router, prefix="/monitoring", tags=["Monitoring"])

    # Global exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions."""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                success=False, message=exc.detail, error_code=f"HTTP_{exc.status_code}"
            ).dict(),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """Handle value errors."""
        logger.error(f"ValueError in {request.url}: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                success=False, message=str(exc), error_code="VALIDATION_ERROR"
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception in {request.url}: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                message="Internal server error" if not debug else str(exc),
                error_code="INTERNAL_ERROR",
            ).dict(),
        )

    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        # Add custom info
        openapi_schema["info"]["x-logo"] = {
            "url": "https://fastapi.tiangolo.com/img/logo-margin/logo-teal.png"
        }

        # Add server information
        openapi_schema["servers"] = [{"url": "/", "description": "Current server"}]

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi

    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "name": app.title,
            "version": app.version,
            "description": app.description,
            "docs_url": "/docs" if debug else None,
            "health_url": "/health",
            "status": "online",
        }

    # Store configuration in app state
    app.state.config = {
        "debug": debug,
        "rate_limiting": enable_rate_limiting,
        "rate_limit_per_minute": rate_limit_per_minute,
        "cors_origins": cors_origins,
    }

    logger.info(f"Created FastAPI app: {title} v{version}")

    return app


def create_production_app() -> FastAPI:
    """Create production-ready FastAPI application."""
    return create_app(
        debug=False,
        enable_rate_limiting=True,
        rate_limit_per_minute=500,  # More conservative for production
        cors_origins=[],  # No CORS in production by default
    )


def create_development_app() -> FastAPI:
    """Create development FastAPI application."""
    return create_app(
        debug=True,
        enable_rate_limiting=False,  # Disabled for development
        cors_origins=["*"],  # Allow all origins in development
    )


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    # Run development server
    uvicorn.run(
        "genesis_humanoid_rl.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
