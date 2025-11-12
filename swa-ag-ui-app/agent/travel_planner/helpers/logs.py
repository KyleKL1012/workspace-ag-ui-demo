import logging
import sys


def get_logger(level: int = logging.INFO, name: str = "travel_planner") -> logging.Logger:
    """
    Configures and returns a logger with the given name.
    Ensures that multiple calls don't duplicate handlers.

    Parameters:
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        name (str): Name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if not logger.hasHandlers():
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger
