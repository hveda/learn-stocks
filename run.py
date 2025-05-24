#!/usr/bin/env python3
"""
Convenience script to run the stock data collection.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

def run_data_collection():
    """Run the Yahoo Finance data collection script with default parameters."""
    from src.data_collection.collect_yahoo_data import main
    sys.argv = [sys.argv[0]]  # Reset argv to avoid conflicts with this script's args
    main()

def run_project_setup():
    """Run the project setup script."""
    from setup_project import main
    main()

def run_data_processing():
    """Run data processing script"""
    from src.data_processing.process_data import process_data
    process_data()

def run_exploratory_analysis():
    """Run exploratory data analysis"""
    from src.exploration.eda_basic_stats import main as run_basic_stats
    from src.exploration.eda_time_series import main as run_time_series
    
    # Create results directory if it doesn't exist
    results_dir = Path(os.path.dirname(__file__)) / 'results' / 'eda'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running basic statistical analysis...")
    run_basic_stats()
    
    print("Running time series analysis...")
    run_time_series()

def run_model_training():
    """Run model training (ARIMA and Prophet)"""
    from src.models.arima_model import run_arima_analysis
    from src.models.prophet_model import run_prophet_analysis
    
    # Tickers to analyze
    tickers = ['BBCA.JK', 'PTBA.JK']
    
    # Create necessary directories
    for model_type in ['arima', 'prophet']:
        model_dir = Path(os.path.dirname(__file__)) / 'models' / model_type
        results_dir = Path(os.path.dirname(__file__)) / 'results' / model_type
        plots_dir = results_dir / 'plots'
        
        model_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Run ARIMA model training and forecasting
    print("Training ARIMA models...")
    for ticker in tickers:
        print(f"Processing {ticker} with ARIMA...")
        run_arima_analysis(ticker, forecast_periods=30)
    
    # Run Prophet model training and forecasting
    print("Training Prophet models...")
    for ticker in tickers:
        print(f"Processing {ticker} with Prophet...")
        run_prophet_analysis(ticker, forecast_periods=30)

def run_reporting():
    """Generate reports and visualizations for model comparison"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from pathlib import Path
    
    print("Generating model comparison reports...")
    
    # Define paths
    results_dir = Path(os.path.dirname(__file__)) / 'results'
    reports_dir = results_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Tickers we're analyzing
    tickers = ['BBCA.JK', 'PTBA.JK']
    
    # Collect metrics from both models
    all_metrics = {}
    
    for ticker in tickers:
        ticker_clean = ticker.replace('.', '_')
        all_metrics[ticker] = {}
        
        # Load ARIMA metrics if available
        arima_metrics_file = results_dir / 'arima' / f"{ticker_clean}_metrics.csv"
        if arima_metrics_file.exists():
            try:
                arima_metrics = pd.read_csv(arima_metrics_file)
                all_metrics[ticker]['ARIMA'] = {
                    'MAE': arima_metrics['MAE'].values[0],
                    'RMSE': arima_metrics['RMSE'].values[0],
                    'MAPE': arima_metrics['MAPE'].values[0],
                    'R2': arima_metrics['R2'].values[0]
                }
                print(f"Loaded ARIMA metrics for {ticker}")
            except Exception as e:
                print(f"Error loading ARIMA metrics for {ticker}: {str(e)}")
        
        # Load Prophet metrics if available
        prophet_metrics_file = results_dir / 'prophet' / f"{ticker_clean}_metrics.csv"
        if prophet_metrics_file.exists():
            try:
                prophet_metrics = pd.read_csv(prophet_metrics_file)
                all_metrics[ticker]['Prophet'] = {
                    'MAE': prophet_metrics['MAE'].values[0],
                    'RMSE': prophet_metrics['RMSE'].values[0],
                    'MAPE': prophet_metrics['MAPE'].values[0],
                    'R2': prophet_metrics['R2'].values[0],
                    'Coverage': prophet_metrics['Coverage'].values[0] if 'Coverage' in prophet_metrics.columns else None
                }
                print(f"Loaded Prophet metrics for {ticker}")
            except Exception as e:
                print(f"Error loading Prophet metrics for {ticker}: {str(e)}")
    
    # Create comparison dataframes and visualizations
    for ticker in tickers:
        if ticker not in all_metrics or not all_metrics[ticker]:
            print(f"No metrics available for {ticker}")
            continue
            
        ticker_metrics = all_metrics[ticker]
        
        # Create comparison dataframe
        models = list(ticker_metrics.keys())
        metrics_df = pd.DataFrame({
            'Model': models,
            'MAE': [ticker_metrics[model].get('MAE') for model in models],
            'RMSE': [ticker_metrics[model].get('RMSE') for model in models],
            'MAPE': [ticker_metrics[model].get('MAPE') for model in models],
            'R2': [ticker_metrics[model].get('R2') for model in models]
        })
        
        # Save comparison to CSV
        ticker_clean = ticker.replace('.', '_')
        comparison_file = reports_dir / f"{ticker_clean}_model_comparison.csv"
        metrics_df.to_csv(comparison_file, index=False)
        print(f"Saved model comparison for {ticker} to {comparison_file}")
        
        # Create comparison visualizations
        plt.figure(figsize=(12, 8))
        
        # Plot MAE, RMSE, MAPE comparison
        plt.subplot(2, 2, 1)
        plt.bar(metrics_df['Model'], metrics_df['MAE'])
        plt.title(f'{ticker} - Mean Absolute Error')
        plt.ylabel('MAE')
        
        plt.subplot(2, 2, 2)
        plt.bar(metrics_df['Model'], metrics_df['RMSE'])
        plt.title(f'{ticker} - Root Mean Squared Error')
        plt.ylabel('RMSE')
        
        plt.subplot(2, 2, 3)
        plt.bar(metrics_df['Model'], metrics_df['MAPE'])
        plt.title(f'{ticker} - Mean Absolute Percentage Error')
        plt.ylabel('MAPE (%)')
        
        plt.subplot(2, 2, 4)
        plt.bar(metrics_df['Model'], metrics_df['R2'])
        plt.title(f'{ticker} - R-squared')
        plt.ylabel('R²')
        
        plt.tight_layout()
        viz_file = reports_dir / f"{ticker_clean}_model_comparison.png"
        plt.savefig(viz_file)
        plt.close()
        print(f"Saved model comparison visualization for {ticker} to {viz_file}")
    
    # Create aggregate report with all tickers and models
    all_results = []
    for ticker, models in all_metrics.items():
        for model_name, metrics in models.items():
            result = {
                'Ticker': ticker,
                'Model': model_name
            }
            result.update(metrics)
            all_results.append(result)
    
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        aggregate_file = reports_dir / "aggregate_model_comparison.csv"
        all_results_df.to_csv(aggregate_file, index=False)
        print(f"Saved aggregate model comparison to {aggregate_file}")
        
        # Generate final report document
        report_file = reports_dir / "model_comparison_report.md"
        with open(report_file, 'w') as f:
            f.write("# Stock Price Forecasting Model Comparison\n\n")
            f.write("## Overview\n\n")
            f.write("This report compares the performance of ARIMA and Prophet forecasting models on Indonesian stock price data.\n\n")
            
            f.write("## Metrics Explanation\n\n")
            f.write("- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values\n")
            f.write("- **RMSE (Root Mean Squared Error)**: Square root of the average squared differences\n")
            f.write("- **MAPE (Mean Absolute Percentage Error)**: Average percentage difference between predicted and actual values\n")
            f.write("- **R² (R-squared)**: Statistical measure of how close the data are to the fitted regression line\n")
            f.write("- **Coverage** (Prophet only): Percentage of actual values falling within the prediction intervals\n\n")
            
            f.write("## Results by Ticker\n\n")
            for ticker in tickers:
                if ticker in all_metrics and all_metrics[ticker]:
                    f.write(f"### {ticker}\n\n")
                    ticker_metrics = all_metrics[ticker]
                    f.write("| Model | MAE | RMSE | MAPE | R² |\n")
                    f.write("|-------|-----|------|------|----|\n")
                    for model, metrics in ticker_metrics.items():
                        f.write(f"| {model} | {metrics.get('MAE', 'N/A'):.4f} | {metrics.get('RMSE', 'N/A'):.4f} | {metrics.get('MAPE', 'N/A'):.2f}% | {metrics.get('R2', 'N/A'):.4f} |\n")
                    f.write("\n")
            
            f.write("## Conclusion\n\n")
            f.write("Based on the metrics above, we can observe the following patterns:\n\n")
            
            # Simple analysis based on the results
            best_models = {}
            for ticker in tickers:
                if ticker in all_metrics and len(all_metrics[ticker]) > 1:
                    arima_mape = all_metrics[ticker].get('ARIMA', {}).get('MAPE', float('inf'))
                    prophet_mape = all_metrics[ticker].get('Prophet', {}).get('MAPE', float('inf'))
                    best_model = 'ARIMA' if arima_mape < prophet_mape else 'Prophet'
                    best_models[ticker] = best_model
                    
                    f.write(f"- For {ticker}, the {best_model} model performs better in terms of MAPE.\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("1. For trading decisions, consider using the model with lower error metrics for each stock.\n")
            f.write("2. Prophet generally provides better uncertainty estimates through prediction intervals.\n")
            f.write("3. ARIMA may be more suitable for stocks with well-defined trends and seasonality.\n")
            f.write("4. Consider ensemble approaches combining both models for potentially improved results.\n")
        
        print(f"Generated detailed model comparison report at {report_file}")
    else:
        print("No metrics data available to generate reports")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Market ML Project Runner")
    parser.add_argument("--setup", action="store_true", help="Run project setup script")
    parser.add_argument("--collect", action="store_true", help="Run data collection")
    parser.add_argument("--process", action="store_true", help="Run data processing")
    parser.add_argument("--explore", action="store_true", help="Run exploratory analysis")
    parser.add_argument("--train", action="store_true", help="Train ML models")
    parser.add_argument("--report", action="store_true", help="Generate reports")
    parser.add_argument("--all", action="store_true", help="Run full workflow")
    
    args = parser.parse_args()
    
    if args.setup or args.all:
        print("Running project setup...")
        run_project_setup()
    
    if args.collect or args.all:
        print("Running data collection...")
        run_data_collection()
    
    if args.process or args.all:
        print("Running data processing...")
        run_data_processing()
        
    if args.explore or args.all:
        print("Running exploratory analysis...")
        run_exploratory_analysis()
        
    if args.train or args.all:
        print("Training ML models...")
        run_model_training()
        
    if args.report or args.all:
        print("Generating reports...")
        run_reporting()
        
    if not any([args.setup, args.collect, args.process, args.explore, args.train, args.report, args.all]):
        parser.print_help()
