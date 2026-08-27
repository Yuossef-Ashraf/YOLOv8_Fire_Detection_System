"""
Logging configuration for YOLOv8 Fire Detection System.
Provides structured logging for detection events, hardware triggers, and performance metrics.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler


class SafeStreamHandler(logging.StreamHandler):
    """Console stream handler that ensures UTF-8 safety across Windows consoles."""
    def emit(self, record):
        try:
            msg = self.format(record)
            msg = msg.replace('🔥', '[FIRE]').replace('💨', '[SMOKE]').replace('⚠️', '[WARN]').replace('✅', '[OK]')
            stream = self.stream
            stream.write(msg + self.terminator)
            stream.flush()
        except Exception:
            self.handleError(record)


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    log_file_name: str = "fire_detection.log"
) -> logging.Logger:
    """
    Configure root application logging with console and rotating file handlers.
    """
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError:
            pass

    logger = logging.getLogger("fire_detection")
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    # Console Handler
    console_handler = SafeStreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    try:
        log_file_path = os.path.join(log_dir, log_file_name)
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not initialize file logger: {e}")

    return logger


logger = setup_logging()
