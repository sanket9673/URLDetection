import logging
import os
import sys

def setup_logger():
    """
    Configures basic logging using logging.basicConfig.
    Logs to both a file and the console.
    """
    # Auto-create missing directories for logs
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    log_file = os.path.join(log_dir, "pipeline.log")
    
    # Configure logging level and format per requirements
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("HybridURLIntelligence")

logger = setup_logger()
