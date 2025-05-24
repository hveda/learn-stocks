#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Cleaning Module

This script handles data cleaning operations for stock market data:
- Handling missing values
- Adjusting timestamps
- Removing duplicates
- Basic validation

The cleaned data is saved to the cleaned/ directory.
"""
import os
import logging
import pandas as pd
from pathlib import Path
import numpy as np

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

# Constants - Define paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
CLEANED_DATA_DIR = PROJECT_ROOT / 'data' / 'cleaned'

def load_raw_data(ticker):
    """
    Load raw stock data from CSV files
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol (e.g., 'BBCA.JK')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the historical data
    """
    ticker_normalized = ticker.replace('.', '_')
    hist_file = RAW_DATA_DIR / f"{ticker_normalized}_historical.csv"
    
    if not hist_file.exists():
        logger.error(f"Historical data file not found for {ticker}: {hist_file}")
        return None
    
    try:
        # Load historical data with proper date parsing
        df = pd.read_csv(hist_file, parse_dates=['Date'], index_col='Date')
        logger.info(f"Successfully loaded {len(df)} records for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None

def clean_data(df, ticker):
    """
    Clean the stock data DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the raw stock data
    ticker : str
        The stock ticker symbol
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the cleaned data
    """
    if df is None or df.empty:
        logger.error(f"No data to clean for {ticker}")
        return None
    
    logger.info(f"Cleaning data for {ticker} with initial shape {df.shape}")
    
    # Make a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # 1. Handle missing values
    # Log missing values before cleaning
    missing_values = df_cleaned.isnull().sum()
    logger.info(f"Missing values before cleaning for {ticker}:\n{missing_values}")
    
    # Forward fill for price data (use previous day's data)
    price_cols = ['Open', 'High', 'Low', 'Close']
    df_cleaned[price_cols] = df_cleaned[price_cols].fillna(method='ffill')
    
    # For remaining missing price data (e.g., at the beginning), use next day's data
    df_cleaned[price_cols] = df_cleaned[price_cols].fillna(method='bfill')
    
    # For Volume, Dividends, and Stock Splits, fill with 0
    fill_zero_cols = ['Volume', 'Dividends', 'Stock Splits']
    df_cleaned[fill_zero_cols] = df_cleaned[fill_zero_cols].fillna(0)
    
    # 2. Remove duplicate dates
    duplicated_dates = df_cleaned.index.duplicated()
    if duplicated_dates.any():
        logger.warning(f"Found {duplicated_dates.sum()} duplicate dates for {ticker}")
        df_cleaned = df_cleaned[~duplicated_dates]
    
    # 3. Sort by date (ensure chronological order)
    df_cleaned = df_cleaned.sort_index()
    
    # 4. Basic validation - check for unrealistic price jumps
    for col in price_cols:
        # Calculate daily percentage change
        pct_change = df_cleaned[col].pct_change().abs()
        
        # Find unrealistic jumps (e.g., >50% in a single day)
        suspicious_jumps = pct_change > 0.5  # 50% threshold
        if suspicious_jumps.any():
            suspicious_dates = df_cleaned.index[suspicious_jumps]
            logger.warning(f"Found suspicious price jumps in {col} for {ticker} on dates: {suspicious_dates}")
            
            # We'll keep the data but log the warning for manual review
    
    # Log cleaning results
    logger.info(f"Data cleaned for {ticker}. Final shape: {df_cleaned.shape}")
    
    return df_cleaned

def save_cleaned_data(df, ticker):
    """
    Save cleaned stock data to CSV file
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the cleaned stock data
    ticker : str
        The stock ticker symbol
    """
    if df is None or df.empty:
        logger.error(f"No cleaned data to save for {ticker}")
        return
    
    # Create directory if it doesn't exist
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save cleaned data
    ticker_normalized = ticker.replace('.', '_')
    output_file = CLEANED_DATA_DIR / f"{ticker_normalized}_cleaned.csv"
    df.to_csv(output_file)
    logger.info(f"Cleaned data saved to {output_file}")

def process_tickers(tickers):
    """
    Process a list of tickers for cleaning
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to process
    """
    logger.info(f"Starting data cleaning process for {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            # Load data
            df = load_raw_data(ticker)
            
            # Clean data
            cleaned_df = clean_data(df, ticker)
            
            # Save cleaned data
            save_cleaned_data(cleaned_df, ticker)
            
        except Exception as e:
            logger.error(f"Error processing {ticker} during cleaning: {str(e)}")
    
    logger.info("Data cleaning process completed")

def main(tickers=None):
    """
    Main function to clean stock data
    
    Parameters:
    -----------
    tickers : list or None
        List of ticker symbols to process. If None, use default IDX tickers.
    """
    # Use default IDX tickers if none provided
    if tickers is None:
        from src.data_collection.collect_yahoo_data import IDX_TICKERS
        tickers = IDX_TICKERS
    
    process_tickers(tickers)

if __name__ == "__main__":
    main()
