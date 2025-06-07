#!/usr/bin/env python3
"""
Script to run model analysis and comparison
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import LOGS_DIR 
from src.models.arima_model import run_arima_analysis
from src.models.prophet_model import run_prophet_analysis
from src.models.ensemble_model import run_ensemble_analysis
from src.analysis.reporting import generate_performance_report
from src.analysis.risk_assessment import assess_risk
from src.analysis.market_alerts import check_market_alerts

# Configure logging
LOGS_DIR.mkdir(exist_ok=True)
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
    logger.info("Starting model analysis and comparison")
    
    # Run different model analyses
    run_arima_analysis()
    run_prophet_analysis()
    run_ensemble_analysis()
    
    # Generate reports
    generate_performance_report()
    
    # Run risk assessment
    assess_risk()
    
    # Check for market alerts
    check_market_alerts()
    
    logger.info("Model analysis completed successfully")

if __name__ == "__main__":
    main()
