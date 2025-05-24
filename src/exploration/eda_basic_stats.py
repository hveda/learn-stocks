#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis Module

This script performs statistical analysis on stock market data:
- Basic statistics (mean, median, std, min, max)
- Distribution analysis
- Correlation analysis
- Visualization of price movements and patterns
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'exploratory_analysis.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants - Define paths
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
FEATURES_DATA_DIR = PROJECT_ROOT / 'data' / 'features'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'eda'
PLOTS_DIR = RESULTS_DIR / 'plots'

def load_features_data(ticker):
    """
    Load data with engineered features from CSV files
    
    Parameters:
    -----------
    ticker : str
        The stock ticker symbol (e.g., 'BBCA.JK')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the data with engineered features
    """
    ticker_normalized = ticker.replace('.', '_')
    features_file = FEATURES_DATA_DIR / f"{ticker_normalized}_features.csv"
    
    if not features_file.exists():
        logger.error(f"Features data file not found for {ticker}: {features_file}")
        return None
    
    try:
        # Load features data with proper date parsing
        df = pd.read_csv(features_file, parse_dates=['Date'], index_col='Date')
        logger.info(f"Successfully loaded {len(df)} records with features for {ticker}")
        return df
    except Exception as e:
        logger.error(f"Error loading features data for {ticker}: {str(e)}")
        return None

def analyze_basic_statistics(df, ticker):
    """
    Analyze basic statistics of the stock data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the basic statistics
    """
    if df is None or df.empty:
        logger.error(f"No data for statistical analysis for {ticker}")
        return None
    
    logger.info(f"Analyzing basic statistics for {ticker}")
    
    # Select key price/volume columns for basic analysis
    key_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return']
    
    try:
        # Calculate basic statistics
        basic_stats = df[key_cols].describe(percentiles=[0.01, 0.05, 0.10, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
        
        # Add additional statistics
        basic_stats.loc['skew'] = df[key_cols].skew()
        basic_stats.loc['kurtosis'] = df[key_cols].kurtosis()
        
        # Calculate annualized return and volatility
        if 'Return' in df.columns:
            trading_days_per_year = 252  # Typical trading days in a year
            basic_stats.loc['ann_return'] = df['Return'].mean() * trading_days_per_year
            basic_stats.loc['ann_volatility'] = df['Return'].std() * np.sqrt(trading_days_per_year)
            basic_stats.loc['sharpe_ratio'] = basic_stats.loc['ann_return']['Return'] / basic_stats.loc['ann_volatility']['Return']
        
        logger.info(f"Basic statistics analysis completed for {ticker}")
        
        # Save basic statistics to CSV
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stats_file = RESULTS_DIR / f"{ticker.replace('.', '_')}_basic_stats.csv"
        basic_stats.to_csv(stats_file)
        logger.info(f"Basic statistics saved to {stats_file}")
        
        return basic_stats
        
    except Exception as e:
        logger.error(f"Error in basic statistics analysis for {ticker}: {str(e)}")
        return None

def analyze_distributions(df, ticker):
    """
    Analyze and visualize distributions of key features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    """
    if df is None or df.empty:
        logger.error(f"No data for distribution analysis for {ticker}")
        return
    
    logger.info(f"Analyzing distributions for {ticker}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Select key columns for distribution analysis
    price_cols = ['Open', 'High', 'Low', 'Close']
    indicator_cols = ['RSI', 'MACD', 'ATR', '%K', '%D']
    return_cols = ['Return', 'Volatility_21']
    
    try:
        # 1. Distribution of price data
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(price_cols):
            plt.subplot(2, 2, i+1)
            sns.histplot(df[col], kde=True)
            plt.title(f"{ticker} - {col} Distribution")
            plt.xlabel(col)
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_price_distributions.png")
        plt.close()
        
        # 2. Distribution of returns
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(return_cols):
            plt.subplot(2, 1, i+1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"{ticker} - {col} Distribution")
            plt.xlabel(col)
            plt.ylabel('Frequency')
            
            # Add normal distribution for comparison
            x = np.linspace(df[col].min(), df[col].max(), 100)
            mean = df[col].mean()
            std = df[col].std()
            plt.plot(x, stats.norm.pdf(x, mean, std) * len(df[col]) * (df[col].max() - df[col].min()) / 50, 
                    'r-', alpha=0.6, label='Normal Distribution')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_return_distributions.png")
        plt.close()
        
        # 3. Distribution of technical indicators
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(indicator_cols):
            if col in df.columns:
                plt.subplot(3, 2, i+1)
                sns.histplot(df[col].dropna(), kde=True)
                plt.title(f"{ticker} - {col} Distribution")
                plt.xlabel(col)
                plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_indicator_distributions.png")
        plt.close()
        
        # 4. QQ-Plot for returns (test for normality)
        plt.figure(figsize=(10, 6))
        stats.probplot(df['Return'].dropna(), dist="norm", plot=plt)
        plt.title(f"{ticker} - Return Distribution Q-Q Plot")
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_return_qqplot.png")
        plt.close()
        
        logger.info(f"Distribution analysis completed for {ticker}")
        
    except Exception as e:
        logger.error(f"Error in distribution analysis for {ticker}: {str(e)}")

def analyze_correlations(df, ticker):
    """
    Analyze correlations between features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    """
    if df is None or df.empty:
        logger.error(f"No data for correlation analysis for {ticker}")
        return
    
    logger.info(f"Analyzing correlations for {ticker}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Select key features for correlation analysis
        price_cols = ['Open', 'High', 'Low', 'Close']
        core_features = price_cols + ['Volume', 'Return', 'OBV', 'RSI', 'MACD', 'ATR', '%K', '%D']
        
        core_features = [col for col in core_features if col in df.columns]
        
        # 2. Compute correlation matrix
        corr_matrix = df[core_features].corr()
        
        # 3. Save correlation matrix to CSV
        corr_file = RESULTS_DIR / f"{ticker.replace('.', '_')}_correlation_matrix.csv"
        corr_matrix.to_csv(corr_file)
        logger.info(f"Correlation matrix saved to {corr_file}")
        
        # 4. Visualize correlation matrix as heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
        plt.title(f"{ticker} - Feature Correlations")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_correlation_heatmap.png")
        plt.close()
        
        # 5. Analyze lagged feature correlations with future returns
        if 'target_return_5d' in df.columns:
            lagged_cols = [col for col in df.columns if 'lag' in col]
            target_cols = [col for col in df.columns if 'target' in col]
            
            if lagged_cols and target_cols:
                selected_cols = lagged_cols[:10] + target_cols  # Limit to first 10 lags to keep plot readable
                lag_corr = df[selected_cols].corr()
                
                # Extract correlations with target variables
                target_corr = lag_corr.loc[target_cols, lagged_cols]
                
                # Plot correlations
                plt.figure(figsize=(14, 8))
                sns.heatmap(target_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
                plt.title(f"{ticker} - Lagged Features vs Future Returns Correlations")
                plt.tight_layout()
                plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_lag_correlation_heatmap.png")
                plt.close()
        
        logger.info(f"Correlation analysis completed for {ticker}")
        
    except Exception as e:
        logger.error(f"Error in correlation analysis for {ticker}: {str(e)}")

def visualize_price_patterns(df, ticker):
    """
    Visualize price movements and patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    """
    if df is None or df.empty:
        logger.error(f"No data for price pattern visualization for {ticker}")
        return
    
    logger.info(f"Visualizing price patterns for {ticker}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Price movement over time with volume
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax1.plot(df.index, df['Close'], label='Close Price', color='blue')
        ax1.set_title(f"{ticker} - Price Movement Over Time")
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot volume as bar chart
        ax2.bar(df.index, df['Volume'], color='gray', alpha=0.5)
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_price_movement.png")
        plt.close()
        
        # 2. Price with MA and Bollinger Bands (last 2 years)
        if all(col in df.columns for col in ['MA20', 'BB_Upper', 'BB_Lower']):
            # Get last 2 years of data
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(years=2)
            df_subset = df[df.index >= start_date]
            
            plt.figure(figsize=(15, 8))
            plt.plot(df_subset.index, df_subset['Close'], label='Close Price', color='blue')
            plt.plot(df_subset.index, df_subset['MA20'], label='20-day MA', color='red')
            plt.plot(df_subset.index, df_subset['MA50'], label='50-day MA', color='orange')
            plt.plot(df_subset.index, df_subset['MA200'], label='200-day MA', color='purple')
            plt.plot(df_subset.index, df_subset['BB_Upper'], label='Upper BB', color='green', linestyle='--')
            plt.plot(df_subset.index, df_subset['BB_Lower'], label='Lower BB', color='green', linestyle='--')
            plt.fill_between(df_subset.index, df_subset['BB_Upper'], df_subset['BB_Lower'], alpha=0.1, color='green')
            
            plt.title(f"{ticker} - Price with Moving Averages and Bollinger Bands (Last 2 Years)")
            plt.ylabel('Price')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_price_with_indicators.png")
            plt.close()
        
        # 3. Candlestick chart for most recent 6 months
        try:
            import mplfinance as mpf
            
            # Get last 6 months of data
            end_date = df.index.max()
            start_date = end_date - pd.DateOffset(months=6)
            df_subset = df[df.index >= start_date].copy()
            
            # Prepare data for mplfinance
            df_subset.index = pd.DatetimeIndex(df_subset.index.date)
            
            # Create candlestick chart
            fig, axes = mpf.plot(
                df_subset,
                type='candle',
                style='yahoo',
                title=f"{ticker} - Candlestick Chart (Last 6 Months)",
                ylabel='Price',
                volume=True,
                figsize=(15, 10),
                returnfig=True
            )
            fig.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_candlestick.png")
            plt.close(fig)
        except (ImportError, Exception) as e:
            logger.warning(f"Could not create candlestick chart: {str(e)}")
        
        logger.info(f"Price pattern visualization completed for {ticker}")
        
    except Exception as e:
        logger.error(f"Error in price pattern visualization for {ticker}: {str(e)}")

def process_tickers(tickers):
    """
    Process a list of tickers for exploratory data analysis
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to process
    """
    logger.info(f"Starting exploratory data analysis for {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            # Load features data
            df = load_features_data(ticker)
            
            if df is not None:
                # Basic statistics analysis
                analyze_basic_statistics(df, ticker)
                
                # Distribution analysis
                analyze_distributions(df, ticker)
                
                # Correlation analysis
                analyze_correlations(df, ticker)
                
                # Price pattern visualization
                visualize_price_patterns(df, ticker)
                
        except Exception as e:
            logger.error(f"Error processing {ticker} during exploratory analysis: {str(e)}")
    
    logger.info("Exploratory data analysis completed")

def main(tickers=None):
    """
    Main function to run exploratory data analysis
    
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
