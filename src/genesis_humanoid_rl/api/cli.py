#!/usr/bin/env python3
"""
CLI runner for Genesis Humanoid RL API.

Provides command-line interface for starting the API server
with various configuration options.
"""

import argparse
import sys
import logging
from pathlib import Path

import uvicorn

from .app import create_app, create_production_app, create_development_app


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("api.log")],
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Genesis Humanoid RL REST API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development server with auto-reload
  python -m genesis_humanoid_rl.api.cli --dev --reload

  # Production server
  python -m genesis_humanoid_rl.api.cli --env production --host 0.0.0.0 --port 8000

  # Custom configuration
  python -m genesis_humanoid_rl.api.cli --cors-origins http://localhost:3000 --rate-limit 100
        """,
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind server to (default: 8000)"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )

    # Environment and debug
    parser.add_argument(
        "--env",
        choices=["development", "production", "custom"],
        default="custom",
        help="Environment preset (default: custom)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable development mode (debug, auto-reload, permissive CORS)",
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload on code changes"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with docs and detailed errors",
    )

    # Security and middleware
    parser.add_argument(
        "--cors-origins",
        nargs="*",
        help="Allowed CORS origins (default: none for production, all for dev)",
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=1000,
        help="Rate limit per client per minute (default: 1000, 0 to disable)",
    )
    parser.add_argument(
        "--api-key-required", action="store_true", help="Require API key authentication"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--access-log", action="store_true", help="Enable access logging"
    )

    # TLS/SSL
    parser.add_argument("--ssl-keyfile", type=Path, help="SSL private key file")
    parser.add_argument("--ssl-certfile", type=Path, help="SSL certificate file")

    # API configuration
    parser.add_argument("--title", default="Genesis Humanoid RL API", help="API title")
    parser.add_argument("--version", default="1.0.0", help="API version")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Create app based on environment
    if args.env == "development" or args.dev:
        logger.info("Creating development app")
        app = create_development_app()
    elif args.env == "production":
        logger.info("Creating production app")
        app = create_production_app()
    else:
        # Custom configuration
        logger.info("Creating custom app")
        app = create_app(
            title=args.title,
            version=args.version,
            debug=args.debug or args.dev,
            enable_rate_limiting=args.rate_limit > 0,
            rate_limit_per_minute=args.rate_limit,
            cors_origins=args.cors_origins,
        )

    # Update app title if provided
    if args.title != "Genesis Humanoid RL API":
        app.title = args.title
    if args.version != "1.0.0":
        app.version = args.version

    # Configure uvicorn settings
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level.lower(),
        "access_log": args.access_log,
        "reload": args.reload or args.dev,
        "workers": 1 if (args.reload or args.dev) else args.workers,
    }

    # Add SSL if provided
    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_config.update(
            {
                "ssl_keyfile": str(args.ssl_keyfile),
                "ssl_certfile": str(args.ssl_certfile),
            }
        )
        logger.info("SSL enabled")

    # Log startup configuration
    logger.info(f"Starting Genesis Humanoid RL API server")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Debug mode: {args.debug or args.dev}")
    logger.info(f"Workers: {uvicorn_config['workers']}")
    logger.info(f"Auto-reload: {uvicorn_config['reload']}")

    if args.cors_origins:
        logger.info(f"CORS origins: {args.cors_origins}")
    if args.rate_limit > 0:
        logger.info(f"Rate limiting: {args.rate_limit} requests/minute")

    try:
        # Start server
        uvicorn.run(**uvicorn_config)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
