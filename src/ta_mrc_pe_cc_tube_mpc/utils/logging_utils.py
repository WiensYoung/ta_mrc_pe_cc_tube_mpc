"""Logging utility using rich for formatted console output."""

import logging
import sys

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


def setup_logger(name: str = "ta_mpc", level: int = logging.INFO) -> logging.Logger:
    """Create a logger with rich handler if available, otherwise standard.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        if _HAS_RICH:
            handler = RichHandler(rich_tracebacks=True)
            formatter = logging.Formatter("%(message)s", datefmt="[%X]")
        else:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_logger(name: str = "ta_mpc") -> logging.Logger:
    """Get or create a module-level logger."""
    return setup_logger(name)
