#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Yahoo Finance Data Collection Module

This script fetches historical stock data from Yahoo Finance for IDX tickers
"""
import os
import time
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data_collection.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_START_DATE = '2000-01-01'
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
IDX_TICKERS = ['BBCA.JK', 'PTBA.JK']  # Bank Central Asia and Bukit Asam
RAW_DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) / 'data' / 'raw'


def fetch_stock_data(ticker, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, sleep_time=1):
    """
    Fetch historical stock data for a given ticker
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    sleep_time : int
        Sleep time in seconds to respect API rate limits
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the historical data
    """
    logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get historical data
        hist_data = stock.history(start=start_date, end=end_date, interval="1d")
        
        if hist_data.empty:
            logger.warning(f"No data returned for {ticker}")
            return None
        
        # Get additional info
        info = stock.info
        
        # Respect API rate limits
        time.sleep(sleep_time)
        
        logger.info(f"Successfully fetched {len(hist_data)} records for {ticker}")
        return hist_data, info
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def save_stock_data(ticker, hist_data, info=None):
    """
    Save stock data to CSV files
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol
    hist_data : pandas.DataFrame
        DataFrame containing the historical data
    info : dict
        Dictionary containing company information
    """
    if hist_data is not None:
        # Create directory if it doesn't exist
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save historical data
        output_file = RAW_DATA_DIR / f"{ticker.replace('.', '_')}_historical.csv"
        hist_data.to_csv(output_file)
        logger.info(f"Historical data saved to {output_file}")
        
        # Save info data if available
        if info is not None:
            info_file = RAW_DATA_DIR / f"{ticker.replace('.', '_')}_info.csv"
            pd.DataFrame(info.items(), columns=['Attribute', 'Value']).to_csv(info_file, index=False)
            logger.info(f"Company info saved to {info_file}")


def main():
    """Main function to fetch and save data for IDX tickers"""
    logger.info("Starting data collection process")
    
    for ticker in IDX_TICKERS:
        try:
            result = fetch_stock_data(ticker)
            if result is not None:
                hist_data, info = result
                save_stock_data(ticker, hist_data, info)
            time.sleep(2)  # Additional sleep between different tickers
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
    
    logger.info("Data collection process completed")


if __name__ == "__main__":
    main()
