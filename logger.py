#!/usr/bin/env python3
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Set up the root logger
logger = logging.getLogger()
logger.setLevel(LOG_LEVEL)

# Configure log format
log_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Add console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_format)
logger.addHandler(console_handler)

# Add file handler for general logs
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "app.log"), maxBytes=10485760, backupCount=10  # 10MB
)
file_handler.setFormatter(log_format)
logger.addHandler(file_handler)

# Add file handler for errors only
error_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "error.log"), maxBytes=10485760, backupCount=10  # 10MB
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(log_format)
logger.addHandler(error_handler)


def get_logger(name):
    """Get a logger with a specific name."""
    return logging.getLogger(name)


class LoggerMiddleware:
    """FastAPI middleware for logging requests and responses."""

    def __init__(self, app):
        self.app = app
        self.logger = get_logger("api.request")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = f"{time.time():.6f}"
        path = scope.get("path", "")
        method = scope.get("method", "")

        # Log the request
        self.logger.info(f"Request {request_id}: {method} {path}")

        # Process request and log any errors
        try:
            await self.app(scope, receive, send)
            self.logger.info(f"Response {request_id}: {method} {path} completed")
        except Exception as e:
            self.logger.error(
                f"Error {request_id}: {method} {path} - {e!s}", exc_info=True
            )
            raise
