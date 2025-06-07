#!/usr/bin/env python3
"""
Script to update the CONSOLIDATED_TECHNICAL_SUMMARY.md document with accurate results from the
actual model outputs.
"""
import os
import sys
import pandas as pd
import re
from pathlib import Path

# Add the project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import (
    PROJECT_ROOT, RESULTS_DIR, ARIMA_RESULTS_DIR, PROPHET_RESULTS_DIR,
    ENSEMBLE_RESULTS_DIR, TICKER_CLEAN
)

def read_model_metrics():
    """Read metrics from the results directory"""
    metrics = {}
    
    # Read Prophet metrics
    for ticker in [TICKER_CLEAN]:
        prophet_metrics_file = PROPHET_RESULTS_DIR / f"{ticker}_metrics.csv"
        if prophet_metrics_file.exists():
            df = pd.read_csv(prophet_metrics_file)
            metrics[f'prophet_{ticker}'] = {
                'MAE': df['MAE'].values[0],
                'RMSE': df['RMSE'].values[0],
                'MAPE': df['MAPE'].values[0],
                'R2': df['R2'].values[0],
                'Coverage': df['Coverage'].values[0] if 'Coverage' in df.columns else None
            }
    
    # Read ARIMA metrics (if they exist)
    for ticker in [TICKER_CLEAN]:
        arima_metrics_file = ARIMA_RESULTS_DIR / f"{ticker}_metrics.csv"
        if arima_metrics_file.exists():
            df = pd.read_csv(arima_metrics_file)
            metrics[f'arima_{ticker}'] = {
                'MAE': df['MAE'].values[0],
                'RMSE': df['RMSE'].values[0],
                'MAPE': df['MAPE'].values[0],
                'R2': df['R2'].values[0],
                'Coverage': df['Coverage'].values[0] if 'Coverage' in df.columns else None
            }
    
    # Read Ensemble metrics (if they exist)
    for ticker in [TICKER_CLEAN]:
        ensemble_metrics_file = ENSEMBLE_RESULTS_DIR / f"{ticker}_metrics.csv"
        if ensemble_metrics_file.exists():
            df = pd.read_csv(ensemble_metrics_file)
            metrics[f'ensemble_{ticker}'] = {
                'MAE': df['MAE'].values[0],
                'RMSE': df['RMSE'].values[0],
                'MAPE': df['MAPE'].values[0],
                'R2': df['R2'].values[0],
                'Coverage': df['Coverage'].values[0] if 'Coverage' in df.columns else None
            }
    
    return metrics

def update_summary_file(metrics):
    """Update the consolidated technical summary file with actual metrics"""
    summary_file = PROJECT_ROOT / "CONSOLIDATED_TECHNICAL_SUMMARY.md"
    
    if not summary_file.exists():
        print(f"Summary file not found: {summary_file}")
        return
    
    with open(summary_file, 'r') as file:
        content = file.read()
    
    # Update ARIMA metrics
    if 'arima_BBCA_JK' in metrics:
        m = metrics['arima_BBCA_JK']
        content = re.sub(r'ARIMA MAE:\s*[\d\.]+', f"ARIMA MAE: {m['MAE']:.4f}", content)
        content = re.sub(r'ARIMA RMSE:\s*[\d\.]+', f"ARIMA RMSE: {m['RMSE']:.4f}", content)
        content = re.sub(r'ARIMA MAPE:\s*[\d\.]+%', f"ARIMA MAPE: {m['MAPE']:.4f}%", content)
        content = re.sub(r'ARIMA R²:\s*[\d\.]+', f"ARIMA R²: {m['R2']:.4f}", content)
    
    # Update Prophet metrics
    if 'prophet_BBCA_JK' in metrics:
        m = metrics['prophet_BBCA_JK']
        content = re.sub(r'Prophet MAE:\s*[\d\.]+', f"Prophet MAE: {m['MAE']:.4f}", content)
        content = re.sub(r'Prophet RMSE:\s*[\d\.]+', f"Prophet RMSE: {m['RMSE']:.4f}", content)
        content = re.sub(r'Prophet MAPE:\s*[\d\.]+%', f"Prophet MAPE: {m['MAPE']:.4f}%", content)
        content = re.sub(r'Prophet R²:\s*[\d\.]+', f"Prophet R²: {m['R2']:.4f}", content)
    
    # Update Ensemble metrics
    if 'ensemble_BBCA_JK' in metrics:
        m = metrics['ensemble_BBCA_JK']
        content = re.sub(r'Ensemble MAE:\s*[\d\.]+', f"Ensemble MAE: {m['MAE']:.4f}", content)
        content = re.sub(r'Ensemble RMSE:\s*[\d\.]+', f"Ensemble RMSE: {m['RMSE']:.4f}", content)
        content = re.sub(r'Ensemble MAPE:\s*[\d\.]+%', f"Ensemble MAPE: {m['MAPE']:.4f}%", content)
        content = re.sub(r'Ensemble R²:\s*[\d\.]+', f"Ensemble R²: {m['R2']:.4f}", content)
    
    # Write the updated content back to the file
    with open(summary_file, 'w') as file:
        file.write(content)
    
    print(f"Updated technical summary: {summary_file}")

def main():
    """Main function to update the technical summary"""
    print("Reading model metrics...")
    metrics = read_model_metrics()
    
    if metrics:
        print(f"Found metrics for {len(metrics)} models.")
        update_summary_file(metrics)
    else:
        print("No metrics found. Make sure models have been trained and evaluated.")

if __name__ == "__main__":
    main()
