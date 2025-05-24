#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature Engineering and Time Series Splitting Module

This script handles:
- Advanced feature engineering for time series forecasting
- Creation of lagged features
- Time series splitting for training/validation/testing

The processed data is saved to the features/ and split/ directories.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import datetime as dt

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
INDICATORS_DATA_DIR = PROJECT_ROOT / 'data' / 'indicators'
FEATURES_DATA_DIR = PROJECT_ROOT / 'data' / 'features'
SPLIT_DATA_DIR = PROJECT_ROOT / 'data' / 'split'

def load_indicators_data(ticker):
    """
    Load data with technical indicators from CSV files
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol (e.g., 'BBCA.JK')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the data with technical indicators
    """
    ticker_normalized = ticker.replace('.', '_')
    indicators_file = INDICATORS_DATA_DIR / f"{ticker_normalized}_indicators.csv"
    
    if not indicators_file.exists():
        logger.error(f"Indicators data file not found for {ticker}: {indicators_file}")
        return None
    
    try:
        # Load indicators data with proper date parsing
        df = pd.read_csv(indicators_file, parse_dates=['Date'], index_col='Date')
        logger.info(f"Successfully loaded {len(df)} records with indicators for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Error loading indicators data for {ticker}: {str(e)}")
        return None

def engineer_features(df, ticker):
    """
    Create advanced features for time series forecasting
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with technical indicators
    ticker : str
        The stock ticker symbol
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional engineered features
    """
    if df is None or df.empty:
        logger.error(f"No data for feature engineering for {ticker}")
        return None
    
    logger.info(f"Engineering features for {ticker}")
    
    # Make a copy to avoid modifying the original
    df_features = df.copy()
    
    # 1. Create lagged features (past n days)
    # Lagged price features
    for i in [1, 2, 3, 5, 10]:
        df_features[f'Close_lag_{i}'] = df_features['Close'].shift(i)
        df_features[f'Volume_lag_{i}'] = df_features['Volume'].shift(i)
        
        # Price changes
        df_features[f'Close_change_{i}'] = df_features['Close'] / df_features[f'Close_lag_{i}'] - 1
    
    # 2. Rolling window statistics
    # Rolling means
    for window in [5, 10, 20]:
        df_features[f'Close_rolling_mean_{window}'] = df_features['Close'].rolling(window=window).mean()
        df_features[f'Volume_rolling_mean_{window}'] = df_features['Volume'].rolling(window=window).mean()
        
        # Rolling standard deviations (volatility)
        df_features[f'Close_rolling_std_{window}'] = df_features['Close'].rolling(window=window).std()
        
        # Rolling min/max
        df_features[f'Close_rolling_min_{window}'] = df_features['Close'].rolling(window=window).min()
        df_features[f'Close_rolling_max_{window}'] = df_features['Close'].rolling(window=window).max()
    
    # 3. Technical indicator cross features
    # Golden/Death Cross (MA50 crossing MA200)
    df_features['golden_cross'] = ((df_features['MA50'] > df_features['MA200']) & 
                                  (df_features['MA50'].shift(1) <= df_features['MA200'].shift(1))).astype(int)
    
    df_features['death_cross'] = ((df_features['MA50'] < df_features['MA200']) & 
                                 (df_features['MA50'].shift(1) >= df_features['MA200'].shift(1))).astype(int)
    
    # MACD signal crosses
    df_features['macd_cross_above'] = ((df_features['MACD'] > df_features['MACD_Signal']) & 
                                      (df_features['MACD'].shift(1) <= df_features['MACD_Signal'].shift(1))).astype(int)
    
    df_features['macd_cross_below'] = ((df_features['MACD'] < df_features['MACD_Signal']) & 
                                     (df_features['MACD'].shift(1) >= df_features['MACD_Signal'].shift(1))).astype(int)
    
    # 4. Price momentum features
    # Rate of change
    for period in [5, 10, 20]:
        df_features[f'roc_{period}'] = df_features['Close'].pct_change(periods=period) * 100
    
    # 5. Volatility features
    # Average True Range ratio
    df_features['atr_ratio'] = df_features['ATR'] / df_features['Close']
    
    # Bollinger Band width
    df_features['bb_width'] = (df_features['BB_Upper'] - df_features['BB_Lower']) / df_features['BB_Middle']
    
    # Bollinger Band position
    df_features['bb_position'] = (df_features['Close'] - df_features['BB_Lower']) / (df_features['BB_Upper'] - df_features['BB_Lower'])
    
    # 6. Calendar features
    # Day of week, month, quarter, year
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    
    # Is month end/start
    df_features['is_month_end'] = df_features.index.is_month_end.astype(int)
    df_features['is_month_start'] = df_features.index.is_month_start.astype(int)
    
    # 7. Target variables for forecasting
    # Next day's return (for classification: up or down)
    df_features['next_day_return'] = df_features['Close'].pct_change(periods=1).shift(-1)
    df_features['next_day_direction'] = np.where(df_features['next_day_return'] > 0, 1, 0)
    
    # Next n-day returns (for regression tasks)
    for days in [1, 3, 5, 10]:
        # Future price
        df_features[f'target_close_{days}d'] = df_features['Close'].shift(-days)
        
        # Future return
        df_features[f'target_return_{days}d'] = df_features['Close'].pct_change(periods=days).shift(-days)
        
        # Future direction
        df_features[f'target_direction_{days}d'] = np.where(df_features[f'target_return_{days}d'] > 0, 1, 0)
        
    # Handle NaN values created during feature engineering
    # Forward fill for price data and technical indicators
    price_cols = ['Open', 'High', 'Low', 'Close', 'MA5', 'MA10', 'MA20', 'MA50', 'MA200',
                  'EMA5', 'EMA10', 'EMA20', 'EMA50', 'EMA200', 'BB_Upper', 'BB_Middle', 'BB_Lower']
    df_features[price_cols] = df_features[price_cols].fillna(method='ffill')
    
    # Drop rows with NaN target values at the end of the dataframe
    # These are rows where we don't have future data to create target variables
    df_features = df_features.dropna(subset=['next_day_return'])
    
    # Drop remaining NaN values that couldn't be filled (typically at the start of the time series)
    df_features = df_features.dropna()
    
    logger.info(f"Feature engineering completed for {ticker}. Shape: {df_features.shape}")
    
    return df_features

def time_series_split(df, ticker, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Perform time series split for training/validation/testing
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with engineered features
    ticker : str
        The stock ticker symbol
    train_size : float
        Proportion of data to use for training
    val_size : float
        Proportion of data to use for validation
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        DataFrames for train, validation, and test sets
    """
    if df is None or df.empty:
        logger.error(f"No data for time series splitting for {ticker}")
        return None, None, None
    
    logger.info(f"Splitting time series data for {ticker}")
    
    # Ensure the DataFrame is sorted by date
    df = df.sort_index()
    
    # Calculate split points
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)
    
    # Split the data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Time series split completed for {ticker}:")
    logger.info(f"Train set: {train_df.shape}, from {train_df.index[0]} to {train_df.index[-1]}")
    logger.info(f"Validation set: {val_df.shape}, from {val_df.index[0]} to {val_df.index[-1]}")
    logger.info(f"Test set: {test_df.shape}, from {test_df.index[0]} to {test_df.index[-1]}")
    
    return train_df, val_df, test_df

def save_features_data(df, ticker):
    """
    Save data with engineered features to CSV file
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data with engineered features
    ticker : str
        The stock ticker symbol
    """
    if df is None or df.empty:
        logger.error(f"No engineered features data to save for {ticker}")
        return
    
    # Create directory if it doesn't exist
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save engineered features data
    ticker_normalized = ticker.replace('.', '_')
    output_file = FEATURES_DATA_DIR / f"{ticker_normalized}_features.csv"
    df.to_csv(output_file)
    logger.info(f"Engineered features data saved to {output_file}")

def save_split_data(train_df, val_df, test_df, ticker):
    """
    Save split data to CSV files
    
    Parameters:
    -----------
    train_df : pandas.DataFrame
        Training data
    val_df : pandas.DataFrame
        Validation data
    test_df : pandas.DataFrame
        Test data
    ticker : str
        The stock ticker symbol
    """
    # Create directory if it doesn't exist
    SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    ticker_normalized = ticker.replace('.', '_')
    
    if train_df is not None and not train_df.empty:
        # Save training data
        train_file = SPLIT_DATA_DIR / f"{ticker_normalized}_train.csv"
        train_df.to_csv(train_file)
        logger.info(f"Training data saved to {train_file}")
    
    if val_df is not None and not val_df.empty:
        # Save validation data
        val_file = SPLIT_DATA_DIR / f"{ticker_normalized}_val.csv"
        val_df.to_csv(val_file)
        logger.info(f"Validation data saved to {val_file}")
    
    if test_df is not None and not test_df.empty:
        # Save test data
        test_file = SPLIT_DATA_DIR / f"{ticker_normalized}_test.csv"
        test_df.to_csv(test_file)
        logger.info(f"Test data saved to {test_file}")

def process_tickers(tickers):
    """
    Process a list of tickers for feature engineering and time series splitting
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to process
    """
    logger.info(f"Starting feature engineering and splitting process for {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            # Load indicators data
            df = load_indicators_data(ticker)
            
            if df is not None:
                # Engineer features
                df_features = engineer_features(df, ticker)
                
                # Save engineered features
                save_features_data(df_features, ticker)
                
                # Split the data
                train_df, val_df, test_df = time_series_split(df_features, ticker)
                
                # Save split data
                save_split_data(train_df, val_df, test_df, ticker)
                
        except Exception as e:
            logger.error(f"Error processing {ticker} during feature engineering/splitting: {str(e)}")
    
    logger.info("Feature engineering and time series splitting process completed")

def main(tickers=None):
    """
    Main function to engineer features and split time series data
    
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
