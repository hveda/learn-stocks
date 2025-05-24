#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data Normalization and Technical Indicators Module

This script handles:
- Data normalization and standardization
- Creation of technical indicators (MA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Feature engineering for time series data

The processed data is saved to the normalized/ and indicators/ directories.
"""
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
CLEANED_DATA_DIR = PROJECT_ROOT / 'data' / 'cleaned'
NORMALIZED_DATA_DIR = PROJECT_ROOT / 'data' / 'normalized'
INDICATORS_DATA_DIR = PROJECT_ROOT / 'data' / 'indicators'

def load_cleaned_data(ticker):
    """
    Load cleaned stock data from CSV files
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol (e.g., 'BBCA.JK')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the cleaned historical data
    """
    ticker_normalized = ticker.replace('.', '_')
    cleaned_file = CLEANED_DATA_DIR / f"{ticker_normalized}_cleaned.csv"
    
    if not cleaned_file.exists():
        logger.error(f"Cleaned data file not found for {ticker}: {cleaned_file}")
        return None
    
    try:
        # Load cleaned data with proper date parsing
        df = pd.read_csv(cleaned_file, parse_dates=['Date'], index_col='Date')
        logger.info(f"Successfully loaded {len(df)} cleaned records for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Error loading cleaned data for {ticker}: {str(e)}")
        return None

def normalize_data(df, ticker):
    """
    Normalize the stock data using Min-Max scaling and standardization
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the cleaned stock data
    ticker : str
        The stock ticker symbol
        
    Returns:
    --------
    tuple
        Two DataFrames - one with min-max scaled data, one with standardized data
    """
    if df is None or df.empty:
        logger.error(f"No data to normalize for {ticker}")
        return None, None
    
    logger.info(f"Normalizing data for {ticker}")
    
    # Make copies to avoid modifying the original
    df_normalized = df.copy()
    df_standardized = df.copy()
    
    # 1. Min-Max scaling (0-1 range)
    price_cols = ['Open', 'High', 'Low', 'Close']
    
    # Create and fit the scalers
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    
    # Apply Min-Max scaling to price columns
    df_normalized[price_cols] = min_max_scaler.fit_transform(df[price_cols])
    
    # Apply standardization (z-score) to price columns
    df_standardized[price_cols] = standard_scaler.fit_transform(df[price_cols])
    
    # 2. Volume normalization - using log transformation for volume (common for highly skewed data)
    if 'Volume' in df.columns:
        # Handle zero values before log transform (add 1)
        df_normalized['Volume'] = np.log1p(df['Volume'])
        df_standardized['Volume'] = np.log1p(df['Volume'])
        
        # Rescale volume after log transform
        vol_minmax = MinMaxScaler()
        vol_standard = StandardScaler()
        
        df_normalized['Volume'] = vol_minmax.fit_transform(df_normalized[['Volume']])
        df_standardized['Volume'] = vol_standard.fit_transform(df_standardized[['Volume']])
    
    # Keep other columns as is (e.g., Dividends, Stock Splits)
    
    logger.info(f"Data normalization completed for {ticker}")
    
    return df_normalized, df_standardized

def calculate_technical_indicators(df, ticker):
    """
    Calculate technical indicators for stock data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the cleaned stock data
    ticker : str
        The stock ticker symbol
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with technical indicators added
    """
    if df is None or df.empty:
        logger.error(f"No data for technical indicator calculation for {ticker}")
        return None
    
    logger.info(f"Calculating technical indicators for {ticker}")
    
    # Make a copy to avoid modifying the original
    df_indicators = df.copy()
    
    # 1. Moving Averages
    # Simple Moving Averages
    df_indicators['MA5'] = df_indicators['Close'].rolling(window=5).mean()
    df_indicators['MA10'] = df_indicators['Close'].rolling(window=10).mean()
    df_indicators['MA20'] = df_indicators['Close'].rolling(window=20).mean()
    df_indicators['MA50'] = df_indicators['Close'].rolling(window=50).mean()
    df_indicators['MA200'] = df_indicators['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df_indicators['EMA5'] = df_indicators['Close'].ewm(span=5, adjust=False).mean()
    df_indicators['EMA10'] = df_indicators['Close'].ewm(span=10, adjust=False).mean()
    df_indicators['EMA20'] = df_indicators['Close'].ewm(span=20, adjust=False).mean()
    df_indicators['EMA50'] = df_indicators['Close'].ewm(span=50, adjust=False).mean()
    df_indicators['EMA200'] = df_indicators['Close'].ewm(span=200, adjust=False).mean()
    
    # 2. Bollinger Bands (20-period SMA with 2 standard deviations)
    ma20 = df_indicators['Close'].rolling(window=20).mean()
    std20 = df_indicators['Close'].rolling(window=20).std()
    df_indicators['BB_Upper'] = ma20 + (std20 * 2)
    df_indicators['BB_Middle'] = ma20
    df_indicators['BB_Lower'] = ma20 - (std20 * 2)
    
    # 3. Relative Strength Index (RSI)
    # Calculate daily price change
    delta = df_indicators['Close'].diff()
    
    # Calculate gains and losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    df_indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. Moving Average Convergence Divergence (MACD)
    ema12 = df_indicators['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_indicators['Close'].ewm(span=26, adjust=False).mean()
    df_indicators['MACD'] = ema12 - ema26
    df_indicators['MACD_Signal'] = df_indicators['MACD'].ewm(span=9, adjust=False).mean()
    df_indicators['MACD_Hist'] = df_indicators['MACD'] - df_indicators['MACD_Signal']
    
    # 5. Stochastic Oscillator
    # %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) × 100
    low_14 = df_indicators['Low'].rolling(window=14).min()
    high_14 = df_indicators['High'].rolling(window=14).max()
    df_indicators['%K'] = ((df_indicators['Close'] - low_14) / (high_14 - low_14)) * 100
    df_indicators['%D'] = df_indicators['%K'].rolling(window=3).mean()
    
    # 6. Average True Range (ATR)
    high_low = df_indicators['High'] - df_indicators['Low']
    high_close_prev = abs(df_indicators['High'] - df_indicators['Close'].shift(1))
    low_close_prev = abs(df_indicators['Low'] - df_indicators['Close'].shift(1))
    
    # Get the maximum of the three
    ranges = pd.concat([high_low, high_close_prev, low_close_prev], axis=1)
    true_range = ranges.max(axis=1)
    df_indicators['ATR'] = true_range.rolling(window=14).mean()
    
    # 7. On-Balance Volume (OBV)
    # Initialize OBV column with zeros
    df_indicators['OBV'] = 0
    
    # Calculate OBV correctly using vectorized operations
    # Get price difference signs
    price_diff = df_indicators['Close'].diff()
    
    # Convert to 1, -1, or 0 based on price movement direction
    signal = np.where(price_diff > 0, 1, np.where(price_diff < 0, -1, 0))
    
    # Multiply signal by volume and calculate cumulative sum
    df_indicators['OBV'] = (signal * df_indicators['Volume']).cumsum()
    
    # 8. Additional price-based features
    # Daily returns
    df_indicators['Return'] = df_indicators['Close'].pct_change()
    
    # Rolling volatility (standard deviation of returns)
    df_indicators['Volatility_21'] = df_indicators['Return'].rolling(window=21).std()
    
    logger.info(f"Technical indicators calculation completed for {ticker}")
    
    return df_indicators

def save_normalized_data(df_minmax, df_zscore, ticker):
    """
    Save normalized data to CSV files
    
    Parameters:
    -----------
    df_minmax : pandas.DataFrame
        DataFrame containing the min-max scaled data
    df_zscore : pandas.DataFrame
        DataFrame containing the standardized (z-score) data
    ticker : str
        The stock ticker symbol
    """
    # Create directory if it doesn't exist
    NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    ticker_normalized = ticker.replace('.', '_')
    
    if df_minmax is not None and not df_minmax.empty:
        # Save min-max normalized data
        minmax_file = NORMALIZED_DATA_DIR / f"{ticker_normalized}_minmax.csv"
        df_minmax.to_csv(minmax_file)
        logger.info(f"Min-max normalized data saved to {minmax_file}")
    
    if df_zscore is not None and not df_zscore.empty:
        # Save z-score standardized data
        zscore_file = NORMALIZED_DATA_DIR / f"{ticker_normalized}_zscore.csv"
        df_zscore.to_csv(zscore_file)
        logger.info(f"Z-score standardized data saved to {zscore_file}")

def save_technical_indicators(df_indicators, ticker):
    """
    Save data with technical indicators to CSV file
    
    Parameters:
    -----------
    df_indicators : pandas.DataFrame
        DataFrame containing the data with technical indicators
    ticker : str
        The stock ticker symbol
    """
    if df_indicators is None or df_indicators.empty:
        logger.error(f"No technical indicators data to save for {ticker}")
        return
    
    # Create directory if it doesn't exist
    INDICATORS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save technical indicators data
    ticker_normalized = ticker.replace('.', '_')
    output_file = INDICATORS_DATA_DIR / f"{ticker_normalized}_indicators.csv"
    df_indicators.to_csv(output_file)
    logger.info(f"Technical indicators data saved to {output_file}")

def process_tickers(tickers):
    """
    Process a list of tickers for normalization and technical indicators
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to process
    """
    logger.info(f"Starting normalization and indicators process for {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            # Load cleaned data
            df = load_cleaned_data(ticker)
            
            if df is not None:
                # Normalize data
                df_minmax, df_zscore = normalize_data(df, ticker)
                
                # Save normalized data
                save_normalized_data(df_minmax, df_zscore, ticker)
                
                # Calculate technical indicators
                df_indicators = calculate_technical_indicators(df, ticker)
                
                # Save data with technical indicators
                save_technical_indicators(df_indicators, ticker)
                
        except Exception as e:
            logger.error(f"Error processing {ticker} during normalization/indicators: {str(e)}")
    
    logger.info("Normalization and technical indicators process completed")

def main(tickers=None):
    """
    Main function to normalize data and calculate technical indicators
    
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
