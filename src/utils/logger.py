"""
logger.py
---------
Provides a named, consistently-formatted logger for the project.
"""

import logging
import sys
from functools import lru_cache


@lru_cache(maxsize=None)
def get_logger(name: str = "crypto_volatility") -> logging.Logger:
    """
    Return a named logger, creating and configuring it on first call.

    Uses an LRU cache so the same Logger instance is reused across imports
    and basicConfig is not called multiple times (which would add duplicate handlers).

    Parameters
    ----------
    name : Logger name (default 'crypto_volatility').

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent log records from propagating to the root logger
        logger.propagate = False

    return logger