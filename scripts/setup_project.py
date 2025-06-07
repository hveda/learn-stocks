#!/usr/bin/env python3
"""
Initialize the project structure and setup configuration files.
"""

import os
import logging
import sys
from pathlib import Path
import importlib.util

# Add the project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

# Import config after adding project root to path
from config.settings import (
    PROJECT_ROOT, DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR,
    NORMALIZED_DATA_DIR, INDICATORS_DATA_DIR, FEATURES_DATA_DIR,
    SPLIT_DATA_DIR, RESULTS_DIR, EDA_RESULTS_DIR, PLOTS_DIR,
    ARIMA_RESULTS_DIR, PROPHET_RESULTS_DIR, ENSEMBLE_RESULTS_DIR,
    REPORTS_DIR, RISK_DIR, ALERTS_DIR, MODELS_DIR, ARIMA_MODELS_DIR,
    PROPHET_MODELS_DIR, ENSEMBLE_MODELS_DIR, LOGS_DIR
)


def setup_logging():
    """
    Configure logging for the project.
    """
    LOGS_DIR.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGS_DIR / "setup.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("project_setup")


def create_directory_structure(logger):
    """
    Create the necessary directories for the project.
    """
    logger.info("Creating directory structure...")
    
    directories = [
        DATA_DIR, RAW_DATA_DIR, CLEANED_DATA_DIR, NORMALIZED_DATA_DIR, 
        INDICATORS_DATA_DIR, FEATURES_DATA_DIR, SPLIT_DATA_DIR,
        RESULTS_DIR, EDA_RESULTS_DIR, PLOTS_DIR, ARIMA_RESULTS_DIR, 
        PROPHET_RESULTS_DIR, ENSEMBLE_RESULTS_DIR, REPORTS_DIR, RISK_DIR, ALERTS_DIR,
        MODELS_DIR, ARIMA_MODELS_DIR, PROPHET_MODELS_DIR, ENSEMBLE_MODELS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def create_init_files(logger):
    """
    Create __init__.py files in the src directories.
    """
    logger.info("Creating __init__.py files...")
    
    src_dir = PROJECT_ROOT / "src"
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py in src directory
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write('# Source package initialization')
        logger.info(f"Created {init_file}")
    
    # Create subdirectories and __init__.py files in src
    subdirectories = [
        "data_collection", 
        "data_processing", 
        "exploration", 
        "analysis", 
        "models", 
        "visualization"
    ]
    
    for subdir in subdirectories:
        subdir_path = src_dir / subdir
        subdir_path.mkdir(exist_ok=True)
        
        # Create __init__.py in subdirectory
        subdir_init = subdir_path / "__init__.py"
        if not subdir_init.exists():
            with open(subdir_init, 'w') as f:
                f.write(f'# {subdir} package initialization')
            logger.info(f"Created {subdir_init}")

def create_env_template(logger):
    """
    Create a template .env file if it doesn't exist
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write("""# Environment variables for the IDX Stock Market Analysis project
# API Configuration
ALPHAVANTAGE_API_KEY=your_alphavantage_key_here

# Database Configuration (if needed)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=idx_stock_data
DB_USER=postgres
DB_PASSWORD=postgres

# Email Notification (if implemented)
EMAIL_HOST=smtp.example.com
EMAIL_PORT=587
EMAIL_USER=user@example.com
EMAIL_PASSWORD=your_email_password
NOTIFICATION_RECIPIENTS=recipient1@example.com,recipient2@example.com
""")
        logger.info(f"Created .env template at {env_file}")
    else:
        logger.info(f".env file already exists at {env_file}")

def main():
    """
    Main function to set up the project structure.
    """
    logger = setup_logging()
    
    logger.info("Starting project setup...")
    try:
        create_directory_structure(logger)
        create_init_files(logger)
        create_env_template(logger)
        logger.info("Project setup completed successfully!")
    except Exception as e:
        logger.error(f"Error during project setup: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
