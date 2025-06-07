#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Series Analysis Module

This script performs time series specific analysis:
- Time series decomposition (trend, seasonality, residuals)
- Stationarity tests
- Autocorrelation analysis
- Anomaly detection
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
from pathlib import Path
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'time_series_analysis.log')),
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

def perform_time_series_decomposition(df, ticker, period=252):
    """
    Decompose time series into trend, seasonality, and residuals
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    period : int
        The period for seasonal decomposition (252 for annual seasonality in daily data)
    """
    if df is None or df.empty:
        logger.error(f"No data for time series decomposition for {ticker}")
        return
    
    logger.info(f"Performing time series decomposition for {ticker}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Resample to weekly data to smooth out noise and reduce computational load
        df_weekly = df['Close'].resample('W').mean()
        
        # 2. Perform seasonal decomposition
        # Set period to 52 for weekly data (52 weeks in a year)
        weekly_period = 52
        
        # Fill any missing values with forward fill to ensure decomposition works
        df_weekly = df_weekly.ffill()
        
        decomposition = seasonal_decompose(df_weekly, model='multiplicative', period=weekly_period)
        
        # 3. Plot decomposition
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 15))
        
        # Original time series
        ax1.plot(df_weekly, label='Original')
        ax1.set_title(f"{ticker} - Original Weekly Closing Price")
        ax1.grid(True, alpha=0.3)
        
        # Trend component
        ax2.plot(decomposition.trend, label='Trend')
        ax2.set_title(f"{ticker} - Trend Component")
        ax2.grid(True, alpha=0.3)
        
        # Seasonal component
        ax3.plot(decomposition.seasonal, label='Seasonality')
        ax3.set_title(f"{ticker} - Seasonal Component")
        ax3.grid(True, alpha=0.3)
        
        # Residual component
        ax4.plot(decomposition.resid, label='Residuals')
        ax4.set_title(f"{ticker} - Residual Component")
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in [ax1, ax2, ax3, ax4]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_time_series_decomposition.png")
        plt.close()
        
        # 4. Save decomposition components
        decomp_data = pd.DataFrame({
            'original': df_weekly,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid
        })
        
        decomp_file = RESULTS_DIR / f"{ticker.replace('.', '_')}_decomposition.csv"
        decomp_data.to_csv(decomp_file)
        logger.info(f"Time series decomposition data saved to {decomp_file}")
        
        logger.info(f"Time series decomposition completed for {ticker}")
        
        return decomposition
        
    except Exception as e:
        logger.error(f"Error in time series decomposition for {ticker}: {str(e)}")
        return None

def test_stationarity(df, ticker):
    """
    Test for stationarity using ADF and KPSS tests
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    """
    if df is None or df.empty:
        logger.error(f"No data for stationarity testing for {ticker}")
        return
    
    logger.info(f"Testing stationarity for {ticker}")
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize results dictionary
        stationarity_results = {}
        
        # Series to test
        test_series = {
            'close': df['Close'],
            'returns': df['Return'].dropna(),
            'log_price': np.log(df['Close']),
            'diff_log_price': np.log(df['Close']).diff().dropna()
        }
        
        # Augmented Dickey-Fuller test (null hypothesis: series has a unit root, i.e., non-stationary)
        for name, series in test_series.items():
            logger.info(f"Testing stationarity of {name} for {ticker}")
            
            # ADF Test
            adf_result = adfuller(series.dropna())
            adf_output = {
                'ADF Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Lags': adf_result[2],
                'Observations': adf_result[3],
                'Critical Values': adf_result[4]
            }
            
            # KPSS Test (null hypothesis: series is stationary)
            try:
                # Run KPSS test with warnings filtered
                with warnings.catch_warnings(record=True) as w:
                    warnings.filterwarnings('always', category=UserWarning)
                    # Change from 'c' (constant) to 'ct' (constant and trend)
                    kpss_result = kpss(series.dropna(), regression='ct')
                    
                    # Check if we got the p-value warning
                    if w and "p-value" in str(w[0].message):
                        logger.info(f"KPSS test for {name} of {ticker}: p-value is smaller than the reported value")
                
                kpss_output = {
                    'KPSS Statistic': kpss_result[0],
                    'p-value': kpss_result[1],
                    'p-value_note': 'May be smaller than reported' if kpss_result[1] == 0.01 else '',
                    'Lags': kpss_result[2],
                    'Critical Values': kpss_result[3]
                }
            except Exception as kpss_error:
                logger.warning(f"KPSS test failed for {name} of {ticker}: {str(kpss_error)}")
                kpss_output = {
                    'Error': str(kpss_error)
                }
            
            # Store results
            stationarity_results[name] = {
                'ADF': adf_output,
                'KPSS': kpss_output
            }
            
            # Log interpretation
            if adf_result[1] < 0.05:
                logger.info(f"ADF Test: {name} for {ticker} is likely stationary (p-value < 0.05)")
            else:
                logger.info(f"ADF Test: {name} for {ticker} is likely non-stationary (p-value >= 0.05)")
            
            if 'p-value' in kpss_output and kpss_output['p-value'] < 0.05:
                logger.info(f"KPSS Test: {name} for {ticker} is likely non-stationary (p-value < 0.05)")
            elif 'p-value' in kpss_output:
                logger.info(f"KPSS Test: {name} for {ticker} is likely stationary (p-value >= 0.05)")
        
        # Save stationarity results to JSON
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)
        
        stationarity_file = RESULTS_DIR / f"{ticker.replace('.', '_')}_stationarity_tests.json"
        with open(stationarity_file, 'w') as f:
            json.dump(stationarity_results, f, cls=NumpyEncoder, indent=4)
        
        logger.info(f"Stationarity test results saved to {stationarity_file}")
        
    except Exception as e:
        logger.error(f"Error in stationarity testing for {ticker}: {str(e)}")

def analyze_autocorrelation(df, ticker, max_lags=40):
    """
    Analyze autocorrelation and partial autocorrelation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    max_lags : int
        Maximum number of lags to analyze
    """
    if df is None or df.empty:
        logger.error(f"No data for autocorrelation analysis for {ticker}")
        return
    
    logger.info(f"Analyzing autocorrelation for {ticker}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Calculate ACF and PACF for returns
        returns = df['Return'].dropna()
        
        # 2. Plot ACF and PACF
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # ACF
        plot_acf(returns, lags=max_lags, ax=ax1, alpha=0.05)
        ax1.set_title(f"{ticker} - Autocorrelation Function (ACF) of Returns")
        
        # PACF
        plot_pacf(returns, lags=max_lags, ax=ax2, alpha=0.05)
        ax2.set_title(f"{ticker} - Partial Autocorrelation Function (PACF) of Returns")
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_autocorrelation.png")
        plt.close()
        
        # 3. Calculate numerical ACF and PACF values
        acf_values = acf(returns, nlags=max_lags, fft=True)
        pacf_values = pacf(returns, nlags=max_lags)
        
        # 4. Save to CSV
        acf_df = pd.DataFrame({
            'lag': range(len(acf_values)),
            'acf': acf_values,
            'pacf': np.concatenate([pacf_values, np.zeros(len(acf_values) - len(pacf_values))])
        })
        
        acf_file = RESULTS_DIR / f"{ticker.replace('.', '_')}_autocorrelation.csv"
        acf_df.to_csv(acf_file, index=False)
        logger.info(f"Autocorrelation data saved to {acf_file}")
        
        logger.info(f"Autocorrelation analysis completed for {ticker}")
        
    except Exception as e:
        logger.error(f"Error in autocorrelation analysis for {ticker}: {str(e)}")

def detect_anomalies(df, ticker, n_std=3):
    """
    Detect anomalies in price and returns data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the stock data with features
    ticker : str
        The stock ticker symbol
    n_std : int
        Number of standard deviations for anomaly detection threshold
    """
    if df is None or df.empty:
        logger.error(f"No data for anomaly detection for {ticker}")
        return
    
    logger.info(f"Detecting anomalies for {ticker}")
    
    # Create plots directory
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Detect anomalies in returns using standard deviation method
        mean_return = df['Return'].mean()
        std_return = df['Return'].std()
        
        upper_thresh = mean_return + n_std * std_return
        lower_thresh = mean_return - n_std * std_return
        
        # Identify anomalies
        anomalies = df[(df['Return'] > upper_thresh) | (df['Return'] < lower_thresh)].copy()
        
        # 2. Plot price with anomalies highlighted
        plt.figure(figsize=(15, 8))
        plt.plot(df.index, df['Close'], label='Close Price', color='blue')
        plt.scatter(anomalies.index, anomalies['Close'], color='red', label=f'Anomalies ({n_std} std)', zorder=5)
        plt.title(f"{ticker} - Price Movement with Anomalies Detected")
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_anomalies.png")
        plt.close()
        
        # 3. Plot returns with anomalies highlighted
        plt.figure(figsize=(15, 8))
        plt.plot(df.index, df['Return'], label='Daily Returns', color='blue', alpha=0.6)
        plt.axhline(y=upper_thresh, color='red', linestyle='--', label=f'Upper Threshold ({n_std} std)')
        plt.axhline(y=lower_thresh, color='red', linestyle='--', label=f'Lower Threshold ({n_std} std)')
        plt.scatter(anomalies.index, anomalies['Return'], color='red', label='Anomalies', zorder=5)
        plt.title(f"{ticker} - Returns with Anomalies Detected")
        plt.ylabel('Daily Return')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{ticker.replace('.', '_')}_return_anomalies.png")
        plt.close()
        
        # 4. Save anomalies to CSV
        anomalies_file = RESULTS_DIR / f"{ticker.replace('.', '_')}_anomalies.csv"
        anomalies.to_csv(anomalies_file)
        logger.info(f"Detected {len(anomalies)} anomalies for {ticker}, saved to {anomalies_file}")
        
        logger.info(f"Anomaly detection completed for {ticker}")
        
        return anomalies
        
    except Exception as e:
        logger.error(f"Error in anomaly detection for {ticker}: {str(e)}")
        return None

def process_tickers(tickers):
    """
    Process a list of tickers for time series analysis
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols to process
    """
    logger.info(f"Starting time series analysis for {len(tickers)} tickers")
    
    for ticker in tickers:
        try:
            # Load features data
            df = load_features_data(ticker)
            
            if df is not None:
                # Time series decomposition
                perform_time_series_decomposition(df, ticker)
                
                # Stationarity tests
                test_stationarity(df, ticker)
                
                # Autocorrelation analysis
                analyze_autocorrelation(df, ticker)
                
                # Anomaly detection
                detect_anomalies(df, ticker)
                
        except Exception as e:
            logger.error(f"Error processing {ticker} during time series analysis: {str(e)}")
    
    logger.info("Time series analysis completed")

def main(tickers=None):
    """
    Main function to run time series analysis
    
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
