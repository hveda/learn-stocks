#!/usr/bin/env python3
"""
Stock data crawler for Yahoo Finance.

This script fetches historical stock data for specified tickers from Yahoo Finance,
handling rate limiting and data validation.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables from .env file
load_dotenv()

# Set up logging
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"data_collection_{timestamp}.log"
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("data_collection")


def fetch_stock_data(ticker, start_date, end_date, logger):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., "BBCA.JK")
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        logger: Logger instance
        
    Returns:
        pandas.DataFrame: Historical stock data
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    try:
        # Respect rate limits by adding a small delay
        time.sleep(1)  
        
        # Fetch data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        # Basic validation
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return None
            
        logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def fetch_company_fundamentals(ticker, logger):
    """
    Fetch fundamental data for a company from Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        logger: Logger instance
        
    Returns:
        dict: Company fundamental data
    """
    logger.info(f"Fetching fundamental data for {ticker}")
    try:
        # Respect rate limits
        time.sleep(1)
        
        stock = yf.Ticker(ticker)
        
        # Collect fundamental data
        info = stock.info
        
        # Basic validation
        if not info or len(info) < 5:  # Arbitrary check that we got some data
            logger.warning(f"Limited or no fundamental data returned for {ticker}")
            return None
            
        logger.info(f"Successfully fetched fundamental data for {ticker}")
        return info
        
    except Exception as e:
        logger.error(f"Error fetching fundamental data for {ticker}: {str(e)}")
        return None


def save_data(data, ticker, data_type, data_dir, logger):
    """
    Save data to CSV file.
    
    Args:
        data: Data to save (DataFrame or dict)
        ticker (str): Stock ticker symbol
        data_type (str): Type of data ('historical' or 'fundamental')
        data_dir (Path): Directory to save data
        logger: Logger instance
    """
    data_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d")
    
    if data_type == 'historical':
        file_path = data_dir / f"{ticker}_historical_{timestamp}.csv"
        data.to_csv(file_path)
        logger.info(f"Historical data saved to {file_path}")
    
    elif data_type == 'fundamental':
        file_path = data_dir / f"{ticker}_fundamental_{timestamp}.csv"
        # Convert dictionary to DataFrame for easy saving
        df = pd.DataFrame([data])
        df.to_csv(file_path, index=False)
        logger.info(f"Fundamental data saved to {file_path}")


def validate_data(df, ticker, logger):
    """
    Validate and clean stock data.
    
    Args:
        df (pandas.DataFrame): Stock data DataFrame
        ticker (str): Stock ticker symbol
        logger: Logger instance
        
    Returns:
        pandas.DataFrame: Cleaned DataFrame
    """
    if df is None:
        return None
        
    logger.info(f"Validating data for {ticker}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values in {ticker} data: {missing[missing > 0]}")
        
    # Check for duplicate indices
    duplicates = df.index.duplicated()
    if duplicates.any():
        logger.warning(f"Found {duplicates.sum()} duplicate timestamps in {ticker} data")
        df = df[~duplicates]
    
    # Check for outliers in closing prices (simple Z-score method)
    z_scores = np.abs((df['Close'] - df['Close'].mean()) / df['Close'].std())
    outliers = z_scores > 3  # Three standard deviations
    if outliers.any():
        logger.warning(f"Found {outliers.sum()} potential outliers in {ticker} closing prices")
    
    return df


def main():
    """Main function to run the data collection."""
    parser = argparse.ArgumentParser(description="Stock data crawler for Yahoo Finance")
    parser.add_argument("--tickers", nargs="+", default=["BBCA.JK", "PTBA.JK"], 
                      help="Stock ticker symbols (default: BBCA.JK PTBA.JK)")
    parser.add_argument("--start-date", default="2000-01-01", 
                      help="Start date in YYYY-MM-DD format (default: 2000-01-01)")
    parser.add_argument("--end-date", default=datetime.now().strftime("%Y-%m-%d"), 
                      help="End date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--include-fundamentals", action="store_true", 
                      help="Also fetch fundamental company data")
    parser.add_argument("--output-dir", default="data/raw", 
                      help="Directory to save data (default: data/raw)")
    parser.add_argument("--log-level", default="INFO", 
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                      help="Set the logging level")
                      
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = setup_logging(log_level)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"Starting data collection for {', '.join(args.tickers)}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    
    # Process each ticker
    for ticker in tqdm(args.tickers, desc="Processing tickers"):
        # Fetch historical data
        df = fetch_stock_data(ticker, args.start_date, args.end_date, logger)
        
        if df is not None:
            # Validate and clean data
            df = validate_data(df, ticker, logger)
            
            if df is not None:
                # Save historical data
                save_data(df, ticker, 'historical', output_dir, logger)
        
        # Fetch fundamental data if requested
        if args.include_fundamentals:
            fundamentals = fetch_company_fundamentals(ticker, logger)
            if fundamentals:
                save_data(fundamentals, ticker, 'fundamental', output_dir, logger)
        
        # Respect rate limits between tickers
        time.sleep(2)
    
    logger.info("Data collection completed")


if __name__ == "__main__":
    main()
