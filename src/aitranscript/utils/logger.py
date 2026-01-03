"""Centralized logging configuration using Rich"""

import logging

from rich.logging import RichHandler


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logger with Rich handler for beautiful console output

    Args:
        name: Logger name (typically __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance with Rich formatting

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.warning("Low memory detected")
        >>> logger.error("Failed to load model")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,  # Beautiful error tracebacks
            markup=True,  # Enable Rich markup in messages
            show_time=True,  # Show timestamps
            show_path=False,  # Hide file paths (cleaner output)
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with default settings

    Convenience function for common use case with INFO level.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Example:
        >>> from aitranscript.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting transcription")
    """
    return setup_logger(name)
