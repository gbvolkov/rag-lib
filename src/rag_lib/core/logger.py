import logging
import sys
from rag_lib.config import settings

def setup_logger(name: str = "rag_lib") -> logging.Logger:
    """
    Configures and returns a logger instance.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    # Set Level
    log_level = settings.log_level.upper()
    logger.setLevel(getattr(logging, log_level, logging.INFO))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

# Global logger instance
logger = setup_logger()
