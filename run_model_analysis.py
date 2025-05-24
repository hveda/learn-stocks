#!/usr/bin/env python3
"""
Script to run model analysis and comparison
"""
import os
import sys
import logging
from pathlib import Path
from run import run_model_training, run_reporting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the model analysis pipeline"""
    logger.info("Starting model analysis and comparison")
    
    # Train the models
    logger.info("Training ARIMA and Prophet models")
    run_model_training()
    
    # Generate reports
    logger.info("Generating comparison reports")
    run_reporting()
    
    logger.info("Model analysis and comparison completed")

if __name__ == "__main__":
    main()
