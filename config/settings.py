#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project Configuration Settings

This module centralizes all configuration parameters for the IDX Stock Market 
Analysis and Prediction project.
"""
import os
from pathlib import Path

# Project root directory (one level up from this file)
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Source directories
SRC_DIR = PROJECT_ROOT / 'src'
VISUALIZATION_DIR = SRC_DIR / 'visualization'
DASHBOARD_SCRIPT = VISUALIZATION_DIR / 'dashboard.py'

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
CLEANED_DATA_DIR = DATA_DIR / 'cleaned'
NORMALIZED_DATA_DIR = DATA_DIR / 'normalized' 
INDICATORS_DATA_DIR = DATA_DIR / 'indicators'
FEATURES_DATA_DIR = DATA_DIR / 'features'
SPLIT_DATA_DIR = DATA_DIR / 'split'

# Results directories
RESULTS_DIR = PROJECT_ROOT / 'results'
EDA_RESULTS_DIR = RESULTS_DIR / 'eda'
PLOTS_DIR = EDA_RESULTS_DIR / 'plots'
ARIMA_RESULTS_DIR = RESULTS_DIR / 'arima'
PROPHET_RESULTS_DIR = RESULTS_DIR / 'prophet'
ENSEMBLE_RESULTS_DIR = RESULTS_DIR / 'ensemble'
REPORTS_DIR = RESULTS_DIR / 'reports'
RISK_DIR = RESULTS_DIR / 'risk_assessment'
ALERTS_DIR = RESULTS_DIR / 'alerts'

# Models directory
MODELS_DIR = PROJECT_ROOT / 'models'
ARIMA_MODELS_DIR = MODELS_DIR / 'arima'
PROPHET_MODELS_DIR = MODELS_DIR / 'prophet'
ENSEMBLE_MODELS_DIR = MODELS_DIR / 'ensemble'

# Logs directory
LOGS_DIR = PROJECT_ROOT / 'logs'

# Ticker settings
DEFAULT_TICKER = 'BBCA.JK'
TICKER_CLEAN = DEFAULT_TICKER.replace('.', '_')

# Yahoo Finance API settings
YAHOO_FINANCE_RATE_LIMIT = 2000  # requests per hour
DEFAULT_START_DATE = '2000-01-01'
DEFAULT_END_DATE = None  # Will be set to current date when needed

# Model parameters
FORECAST_PERIODS = 30
