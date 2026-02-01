"""Structured logging with optional trace_id."""

import logging
import sys
from typing import Optional

from app.core.config import get_settings


def get_logger(name: str, trace_id: Optional[str] = None) -> logging.Logger:
    """Return a logger with optional trace_id in extra for observability."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if trace_id:
        logger = logging.LoggerAdapter(logger, extra={"trace_id": trace_id})
    return logger
