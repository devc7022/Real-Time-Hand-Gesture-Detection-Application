import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = "INFO") -> logging.Logger:
    """
    Configures stream logging with standard formatters.
    Prevents duplicate handler registrations.
    """
    level = getattr(logging, log_level.upper() if log_level else "INFO", logging.INFO)
    
    root_logger = logging.getLogger("hand_gesture_app")
    root_logger.setLevel(level)

    if not root_logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger


# Default logger for utility calls
logger = setup_logging()
