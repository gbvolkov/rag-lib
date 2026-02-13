import pytest
import logging
from io import StringIO
from rag_lib.core.logger import setup_logger

def test_logger_capture():
    # Setup
    stream = StringIO()
    # Use a unique name to avoid global state pollution
    logger = logging.getLogger("test_logger_unique")
    logger.setLevel(logging.INFO) # Force level
    
    # Add a stream handler to capture output for this test
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Act
    logger.info("Test Info Message")
    logger.warning("Test Warning Message")
    
    # Assert
    log_output = stream.getvalue()
    print(f"DEBUG: Log Output: {log_output}") # For debug if fails
    assert "INFO - Test Info Message" in log_output
    assert "WARNING - Test Warning Message" in log_output
    
    # Cleanup
    logger.removeHandler(handler)
