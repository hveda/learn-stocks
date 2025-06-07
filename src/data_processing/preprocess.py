#!/usr/bin/env python3
"""
Stock data preprocessing module.

This module handles missing values, normalization, and standardization of stock data.
"""
import os
import logging
from pathlib import Path
import pandas as pd
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

# Define paths
PROJECT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
RAW_DATA_DIR = PROJECT_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = PROJECT_DIR / 'data' / 'processed'


def load_stock_data(ticker):
    """
    Load stock data from raw data directory.
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol without dots (e.g., 'BBCA_JK')
        
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing the historical stock data or None if not found
    """
    try:
        file_path = RAW_DATA_DIR / f"{ticker}_historical.csv"
        if not file_path.exists():
            logger.warning(f"No data file found for {ticker} at {file_path}")
            return None
        
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        logger.info(f"Loaded {len(df)} records for {ticker}")
        return df
    
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None


def handle_missing_values(df, method='ffill'):
    """
    Handle missing values in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data
    method : str
        Method to fill missing values ('ffill', 'bfill', 'linear', 'mean')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values handled
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe provided to handle_missing_values")
        return df
    
    # Count missing values before treatment
    missing_before = df.isna().sum()
    if missing_before.sum() > 0:
        logger.info(f"Missing values before treatment: \n{missing_before}")
    
    # Handle missing values based on method
    try:
        if method == 'ffill':
            df = df.ffill()
            # If there are still NaNs at the beginning, fill them with bfill
            df = df.bfill()
        elif method == 'bfill':
            df = df.bfill()
            df = df.ffill()  # Handle beginning values
        elif method == 'linear':
            df = df.interpolate(method='linear')
            df = df.ffill().bfill()  # Handle edges
        elif method == 'mean':
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        else:
            logger.warning(f"Unsupported missing value handling method: {method}. Using ffill instead.")
            df = df.ffill().bfill()
    
        # Count missing values after treatment
        missing_after = df.isna().sum()
        if missing_after.sum() > 0:
            logger.warning(f"There are still missing values after treatment: \n{missing_after}")
        else:
            logger.info("All missing values have been handled")
            
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
    
    return df


def normalize_data(df, method='minmax', columns=None):
    """
    Normalize or standardize data in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data
    method : str
        Method to normalize/standardize ('minmax', 'zscore', 'log', 'pct_change')
    columns : list or None
        List of columns to normalize. If None, normalize all numeric columns.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized data and original data preserved
    """
    if df is None or df.empty:
        logger.warning("Empty dataframe provided to normalize_data")
        return df
    
    # Make a copy to avoid modifying the original dataframe
    df_norm = df.copy()
    
    # Determine which columns to normalize
    if columns is None:
        # Select only numeric columns
        columns = df.select_dtypes(include=['number']).columns.tolist()
    
    try:
        if method == 'minmax':
            # Min-max scaling (0 to 1)
            for col in columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                if max_val > min_val:  # Prevent division by zero
                    df_norm[f"{col}_norm"] = (df_norm[col] - min_val) / (max_val - min_val)
                else:
                    df_norm[f"{col}_norm"] = 0
                    logger.warning(f"Column {col} has identical min and max values. Setting normalized values to 0.")
        
        elif method == 'zscore':
            # Z-score standardization
            for col in columns:
                mean_val = df_norm[col].mean()
                std_val = df_norm[col].std()
                if std_val > 0:  # Prevent division by zero
                    df_norm[f"{col}_norm"] = (df_norm[col] - mean_val) / std_val
                else:
                    df_norm[f"{col}_norm"] = 0
                    logger.warning(f"Column {col} has zero standard deviation. Setting standardized values to 0.")
        
        elif method == 'log':
            # Log transformation (useful for right-skewed data)
            for col in columns:
                # Ensure all values are positive
                min_val = df_norm[col].min()
                if min_val <= 0:
                    shift = abs(min_val) + 1
                    df_norm[f"{col}_norm"] = np.log(df_norm[col] + shift)
                    logger.info(f"Column {col} contains non-positive values. Shifting by {shift} before log transform.")
                else:
                    df_norm[f"{col}_norm"] = np.log(df_norm[col])
        
        elif method == 'pct_change':
            # Percentage change (useful for stock prices)
            for col in columns:
                df_norm[f"{col}_pct"] = df_norm[col].pct_change().fillna(0)
        
        else:
            logger.warning(f"Unsupported normalization method: {method}. No normalization performed.")
        
        logger.info(f"Data normalized using {method} method for columns: {columns}")
        
    except Exception as e:
        logger.error(f"Error normalizing data: {str(e)}")
    
    return df_norm


def save_processed_data(df, ticker):
    """
    Save processed data to the processed data directory.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing processed stock data
    ticker : str
        The stock ticker symbol without dots (e.g., 'BBCA_JK')
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    if df is None or df.empty:
        logger.warning(f"No data to save for {ticker}")
        return False
    
    try:
        # Create processed data directory if it doesn't exist
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_file = PROCESSED_DATA_DIR / f"{ticker}_processed.csv"
        df.to_csv(output_file)
        
        logger.info(f"Processed data saved to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving processed data for {ticker}: {str(e)}")
        return False


def process_stock_data(ticker, missing_method='ffill', normalize_method='minmax'):
    """
    Main function to process stock data.
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol with or without dots (e.g., 'BBCA.JK' or 'BBCA_JK')
    missing_method : str
        Method to fill missing values
    normalize_method : str
        Method to normalize data
        
    Returns:
    --------
    pandas.DataFrame
        Processed DataFrame
    """
    # Convert ticker format if needed (replace dots with underscores)
    ticker_file = ticker.replace('.', '_')
    
    logger.info(f"Processing stock data for {ticker}")
    
    # Load data
    df = load_stock_data(ticker_file)
    if df is None:
        return None
    
    # Handle missing values
    df = handle_missing_values(df, method=missing_method)
    
    # Normalize data
    df = normalize_data(df, method=normalize_method, 
                        columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Save processed data
    save_processed_data(df, ticker_file)
    
    return df


def main():
    """
    Process all available stock data in the raw data directory.
    """
    logger.info("Starting data preprocessing")
    
    # Get all raw historical data files
    try:
        raw_files = list(RAW_DATA_DIR.glob('*_historical.csv'))
        tickers = [file.name.replace('_historical.csv', '') for file in raw_files]
        
        if not tickers:
            logger.warning(f"No raw data files found in {RAW_DATA_DIR}")
            return
        
        logger.info(f"Found {len(tickers)} stocks to process: {tickers}")
        
        for ticker in tickers:
            process_stock_data(ticker)
        
        logger.info("Data preprocessing completed")
        
    except Exception as e:
        logger.error(f"Error in preprocessing main function: {str(e)}")


if __name__ == "__main__":
    main()
