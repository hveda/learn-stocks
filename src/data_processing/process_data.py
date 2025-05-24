#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main Data Processing Module

This module orchestrates the entire data processing pipeline:
1. Data cleaning
2. Normalization
3. Technical indicator calculation
4. Feature engineering
5. Time series splitting

It serves as the entry point for the data processing workflow.
"""
import os
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_processing.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import processing modules
from src.data_processing.data_cleaner import main as run_data_cleaning
from src.data_processing.technical_indicators import main as run_technical_indicators
from src.data_processing.feature_engineering import main as run_feature_engineering

def process_data(tickers=None, steps=None):
    """
    Run the full data processing pipeline
    
    Parameters:
    -----------
    tickers : list or None
        List of ticker symbols to process. If None, use default IDX tickers.
    steps : list or None
        List of processing steps to run. If None, run all steps.
        Valid steps are: 'clean', 'indicators', 'features'
    """
    # Use default IDX tickers if none provided
    if tickers is None:
        from src.data_collection.collect_yahoo_data import IDX_TICKERS
        tickers = IDX_TICKERS
        
    # Use all steps if none specified
    all_steps = ['clean', 'indicators', 'features']
    steps = steps if steps is not None else all_steps
    
    logger.info(f"Starting data processing pipeline for {len(tickers)} tickers with steps: {steps}")
    
    try:
        # Step 1: Data Cleaning
        if 'clean' in steps:
            logger.info("Step 1: Data Cleaning")
            try:
                run_data_cleaning(tickers)
                logger.info("Data cleaning completed successfully")
            except Exception as e:
                logger.error(f"Error during data cleaning: {str(e)}")
                if 'indicators' in steps or 'features' in steps:
                    logger.warning("Continuing to next steps despite cleaning errors")
                    
        # Step 2: Technical Indicators
        if 'indicators' in steps:
            logger.info("Step 2: Normalization and Technical Indicators")
            try:
                run_technical_indicators(tickers)
                logger.info("Technical indicators calculation completed successfully")
            except Exception as e:
                logger.error(f"Error during technical indicators calculation: {str(e)}")
                if 'features' in steps:
                    logger.warning("Continuing to next step despite indicators errors")
                    
        # Step 3: Feature Engineering and Time Series Splitting
        if 'features' in steps:
            logger.info("Step 3: Feature Engineering and Time Series Splitting")
            try:
                run_feature_engineering(tickers)
                logger.info("Feature engineering and splitting completed successfully")
            except Exception as e:
                logger.error(f"Error during feature engineering and splitting: {str(e)}")
                
        logger.info("Data processing pipeline completed")
        
    except Exception as e:
        logger.error(f"Unexpected error during data processing pipeline: {str(e)}")
        raise

def main():
    """
    Main function to run the data processing pipeline with command-line arguments
    """
    parser = argparse.ArgumentParser(description="Stock Market Data Processing Pipeline")
    
    parser.add_argument("--tickers", nargs="+", help="Stock ticker symbols to process")
    parser.add_argument("--steps", nargs="+", choices=['clean', 'indicators', 'features'], 
                        help="Processing steps to run")
    
    args = parser.parse_args()
    
    # Run data processing with specified tickers and steps
    process_data(tickers=args.tickers, steps=args.steps)

if __name__ == "__main__":
    main()
