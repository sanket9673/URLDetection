import os
import yaml
from src.logger_config import logger

def load_config(config_path):
    """
    Loads YAML configuration file. Raises an error if missing.
    """
    logger.info("Step start: Loading configuration")
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Missing configuration file at: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Configuration loaded successfully from {config_path}")
        logger.info("Step end: Loading configuration")
        return config
    except FileNotFoundError as e:
        logger.error(f"Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while loading config: {str(e)}")
        raise

def ensure_directory_exists(dir_path):
    """
    Auto-creates directory if it doesn't exist.
    """
    logger.info(f"Step start: Directory check for '{dir_path}'")
    try:
        if not os.path.exists(dir_path):
            logger.info(f"Directory missing. Auto-creating folder: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Folder '{dir_path}' created successfully.")
        else:
            logger.info(f"Directory already exists: {dir_path}")
        logger.info(f"Step end: Directory check for '{dir_path}'")
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {str(e)}")
        raise

def mock_data_pipeline():
    """
    Mocks loading data to demonstrate dataset shape logging.
    """
    logger.info("Step start: Data loading pipeline")
    try:
        dataset_shape = (50000, 256)
        logger.info(f"Dataset shape: {dataset_shape}")
        logger.info("Step end: Data loading pipeline")
        return dataset_shape
    except Exception as e:
        logger.error(f"Error in data pipeline: {str(e)}")
        raise

def mock_training_pipeline(dataset_shape):
    """
    Mocks model training to demonstrate training metrics logging.
    """
    logger.info("Step start: Hybrid Model Training")
    try:
        if not dataset_shape:
            raise ValueError("Invalid dataset shape provided for training.")
            
        training_metrics = {
            "lightgbm_auc": 0.941,
            "gnn_auc": 0.963,
            "hybrid_auc": 0.985,
            "hybrid_loss": 0.098
        }
        logger.info(f"Training metrics: {training_metrics}")
        logger.info("Step end: Hybrid Model Training")
        return training_metrics
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise
