"""
Centralized logging factory.

Design decision: One logger factory used by every package. The duplicate-
handler guard ensures importing the same module twice does not double-log.
Log files rotate daily so each run day has its own file.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a named logger writing to console (INFO+) and file (DEBUG+).

    Args:
        name: Caller passes __name__ so log lines show the exact module.
        log_dir: Directory for daily rotating log files.

    Returns:
        Configured Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # guard against duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)

    log_file = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d") + ".log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger
