#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to check metrics from model results
"""
import pandas as pd
import os
import sys
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))

# Import configuration
from config.settings import ENSEMBLE_RESULTS_DIR, TICKER_CLEAN

def main():
    """Display metrics for the ensemble model"""
    metrics_file = ENSEMBLE_RESULTS_DIR / f"{TICKER_CLEAN}_metrics.csv"
    if not metrics_file.exists():
        print(f"Metrics file not found: {metrics_file}")
        return
        
    metrics = pd.read_csv(metrics_file)
    print(f"Ensemble Metrics for {TICKER_CLEAN.replace('_', '.')}:")
    print(metrics)

if __name__ == "__main__":
    main()
