"""
Logging configuration for DefectDetector service.
"""

import logging
import sys
from typing import Optional

from app.config import settings


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """
    Configure and return the application logger.
    
    Args:
        level: Optional log level override
        
    Returns:
        Configured logger instance
    """
    log_level = level or ("DEBUG" if settings.DEBUG else "INFO")
    
    # Create logger
    logger = logging.getLogger(settings.SERVICE_NAME)
    logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Create default logger instance
logger = setup_logging()

