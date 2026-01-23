"""
Structured logging configuration for Sentinel API.

Configures structlog with JSON formatting for production environments
and human-readable output for development.
"""

import logging
import os
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor


def add_severity_level(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """
    Add severity level compatible with Google Cloud Logging.

    Maps Python log levels to GCP severity levels.
    """
    level_map = {
        "debug": "DEBUG",
        "info": "INFO",
        "warning": "WARNING",
        "error": "ERROR",
        "critical": "CRITICAL",
        "exception": "ERROR",
    }
    event_dict["severity"] = level_map.get(method_name, "DEFAULT")
    return event_dict


def add_service_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add service context for log correlation."""
    event_dict["service"] = "sentinel-api"
    event_dict["version"] = os.getenv("APP_VERSION", "1.0.0")
    return event_dict


def drop_color_message_key(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Remove color_message key used by uvicorn."""
    event_dict.pop("color_message", None)
    return event_dict


def setup_logging(
    log_level: str | None = None,
    json_format: bool | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   Defaults to LOG_LEVEL env var or INFO.
        json_format: Whether to use JSON formatting.
                     Defaults to True in production (LOG_FORMAT=json),
                     False in development.

    Example:
        >>> setup_logging()
        >>> logger = structlog.get_logger()
        >>> logger.info("request_received", path="/health", method="GET")
    """
    # Determine configuration from environment
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    if json_format is None:
        log_format = os.getenv("LOG_FORMAT", "json" if _is_production() else "console")
        json_format = log_format.lower() == "json"

    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level, logging.INFO)

    # Shared processors for all configurations
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        add_severity_level,
        add_service_context,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.ExtraAdder(),
        drop_color_message_key,
    ]

    if json_format:
        # Production: JSON output
        processors: list[Processor] = [
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    else:
        # Development: colored console output
        processors = [
            *shared_processors,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    # Configure uvicorn loggers to use structlog
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.addHandler(handler)
        uvicorn_logger.propagate = False

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _is_production() -> bool:
    """Check if running in production environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    return env in ("production", "prod", "staging")


def get_request_logger(
    request_id: str | None = None,
    user_id: str | None = None,
    **extra: Any,
) -> structlog.stdlib.BoundLogger:
    """
    Get a logger with request context bound.

    Args:
        request_id: Unique request identifier.
        user_id: Authenticated user identifier.
        **extra: Additional context to bind.

    Returns:
        BoundLogger with context bound.

    Example:
        >>> logger = get_request_logger(request_id="abc123", user_id="user@example.com")
        >>> logger.info("processing_event", event_type="auth")
    """
    logger = structlog.get_logger()

    context: dict[str, Any] = {}
    if request_id:
        context["request_id"] = request_id
    if user_id:
        context["user_id"] = user_id
    context.update(extra)

    return logger.bind(**context)


class LoggingMiddleware:
    """
    ASGI middleware for request/response logging.

    Logs incoming requests and outgoing responses with timing information.
    """

    def __init__(self, app: Any) -> None:
        """
        Initialize the middleware.

        Args:
            app: ASGI application.
        """
        self.app = app
        self.logger = structlog.get_logger()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        """Process ASGI request."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        import time
        import uuid

        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Extract request info
        method = scope.get("method", "UNKNOWN")
        path = scope.get("path", "/")
        client = scope.get("client", ("unknown", 0))
        client_ip = client[0] if client else "unknown"

        # Log request
        self.logger.info(
            "request_started",
            request_id=request_id,
            method=method,
            path=path,
            client_ip=client_ip,
        )

        # Track response status
        status_code = 500

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 500)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            self.logger.exception(
                "request_failed",
                request_id=request_id,
                error=str(e),
            )
            raise
        finally:
            duration = time.time() - start_time
            self.logger.info(
                "request_completed",
                request_id=request_id,
                method=method,
                path=path,
                status_code=status_code,
                duration_ms=round(duration * 1000, 2),
            )
