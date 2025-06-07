#!/usr/bin/env python3
"""
Script to run model analysis and comparison
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))

# Import configuration
from config.settings import LOGS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / "model_analysis.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run the model analysis pipeline"""
    from run import (
        run_model_training,
        run_reporting,
        run_risk_assessment,
        run_market_alerts
    )
    
    logger.info("Starting model analysis and comparison")
    
    # Train models
    run_model_training()
    
    # Generate reports
    run_reporting()
    
    # Optional: Run risk assessment and market alerts
    # run_risk_assessment()
    # run_market_alerts()
    
    logger.info("Model analysis complete. Check results in the results directory.")

if __name__ == "__main__":
    main()
