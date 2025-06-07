#!/usr/bin/env python3
"""
Interactive Dashboard for Stock Price Forecasting Models

This script creates an interactive dashboard using Plotly Dash
to visualize and compare the forecasting results from different models.
"""
import os
import sys
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import logging

# Add the project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import settings
from config.settings import (
    PROJECT_ROOT, RESULTS_DIR, ARIMA_RESULTS_DIR, PROPHET_RESULTS_DIR,
    ENSEMBLE_RESULTS_DIR, REPORTS_DIR, EDA_RESULTS_DIR,
    DEFAULT_TICKER, TICKER_CLEAN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'dashboard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define the app
app = dash.Dash(__name__, title="IDX Stock Analysis Dashboard")

# Define available tickers (for now, just BBCA)
TICKERS = [DEFAULT_TICKER]  # Can be expanded later

# Function to load stock data and forecasts
def load_data(ticker):
    """Load historical data and forecasts for the specified ticker"""
    try:
        ticker_clean = ticker.replace('.', '_')
        
        # Load actual historical data
        actual_data = None
        for data_dir in [ARIMA_RESULTS_DIR, PROPHET_RESULTS_DIR, ENSEMBLE_RESULTS_DIR]:
            data_file = data_dir / f"{ticker_clean}_data.csv"
            if data_file.exists():
                actual_data = pd.read_csv(data_file)
                actual_data['Date'] = pd.to_datetime(actual_data['Date'])
                break
        
        if actual_data is None:
            logger.error(f"No historical data found for {ticker}")
            return None, {}, {}
        
        # Load forecasts
        forecasts = {}
        metrics = {}
        
        # Load ARIMA forecast
        arima_forecast_file = ARIMA_RESULTS_DIR / f"{ticker_clean}_forecast.csv"
        if arima_forecast_file.exists():
            arima_forecast = pd.read_csv(arima_forecast_file)
            arima_forecast['Date'] = pd.to_datetime(arima_forecast['Date'])
            forecasts['ARIMA'] = arima_forecast
            
            # Load ARIMA metrics
            arima_metrics_file = ARIMA_RESULTS_DIR / f"{ticker_clean}_metrics.csv"
            if arima_metrics_file.exists():
                metrics['ARIMA'] = pd.read_csv(arima_metrics_file)
        
        # Load Prophet forecast
        prophet_forecast_file = PROPHET_RESULTS_DIR / f"{ticker_clean}_forecast.csv"
        if prophet_forecast_file.exists():
            prophet_forecast = pd.read_csv(prophet_forecast_file)
            prophet_forecast['Date'] = pd.to_datetime(prophet_forecast['Date'])
            if 'ds' in prophet_forecast.columns and 'Date' not in prophet_forecast.columns:
                prophet_forecast = prophet_forecast.rename(columns={'ds': 'Date', 'yhat': 'Forecast'})
            forecasts['Prophet'] = prophet_forecast
            
            # Load Prophet metrics
            prophet_metrics_file = PROPHET_RESULTS_DIR / f"{ticker_clean}_metrics.csv"
            if prophet_metrics_file.exists():
                metrics['Prophet'] = pd.read_csv(prophet_metrics_file)
        
        # Load Ensemble forecast
        ensemble_forecast_file = ENSEMBLE_RESULTS_DIR / f"{ticker_clean}_forecast.csv"
        if ensemble_forecast_file.exists():
            ensemble_forecast = pd.read_csv(ensemble_forecast_file)
            ensemble_forecast['Date'] = pd.to_datetime(ensemble_forecast['Date'])
            forecasts['Ensemble'] = ensemble_forecast
            
            # Load Ensemble metrics
            ensemble_metrics_file = ENSEMBLE_RESULTS_DIR / f"{ticker_clean}_metrics.csv"
            if ensemble_metrics_file.exists():
                metrics['Ensemble'] = pd.read_csv(ensemble_metrics_file)
        
        return actual_data, forecasts, metrics
    
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return None, {}, {}

def create_forecast_plot(ticker, actual_data, forecasts, timeframe='3M'):
    """Create the forecast comparison plot"""
    fig = go.Figure()
    
    # Determine date range based on timeframe
    end_date = pd.Timestamp.now()
    if timeframe == '1M':
        start_date = end_date - pd.DateOffset(months=1)
    elif timeframe == '3M':
        start_date = end_date - pd.DateOffset(months=3)
    elif timeframe == '6M':
        start_date = end_date - pd.DateOffset(months=6)
    elif timeframe == '1Y':
        start_date = end_date - pd.DateOffset(years=1)
    elif timeframe == '3Y':
        start_date = end_date - pd.DateOffset(years=3)
    elif timeframe == '5Y':
        start_date = end_date - pd.DateOffset(years=5)
    else:  # All data
        if actual_data is not None and len(actual_data) > 0:
            start_date = actual_data['Date'].min()
        else:
            start_date = end_date - pd.DateOffset(years=5)
    
    # Filter actual data based on timeframe
    if actual_data is not None and len(actual_data) > 0:
        filtered_actual = actual_data[actual_data['Date'] >= start_date]
        fig.add_trace(go.Scatter(
            x=filtered_actual['Date'],
            y=filtered_actual['Close'],
            mode='lines',
            name='Actual',
            line=dict(color='black')
        ))
    
    # Add forecasts to plot
    colors = {'ARIMA': 'blue', 'Prophet': 'red', 'Ensemble': 'green'}
    for model_name, forecast_df in forecasts.items():
        if forecast_df is not None and len(forecast_df) > 0:
            filtered_forecast = forecast_df[forecast_df['Date'] >= start_date]
            
            # Handle different column names for different models
            forecast_col = 'Forecast'
            lower_col = 'Lower'
            upper_col = 'Upper'
            
            if model_name == 'Prophet' and 'yhat' in forecast_df.columns:
                forecast_col = 'yhat'
                lower_col = 'yhat_lower'
                upper_col = 'yhat_upper'
            
            # Add forecast line
            fig.add_trace(go.Scatter(
                x=filtered_forecast['Date'],
                y=filtered_forecast[forecast_col],
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color=colors.get(model_name, 'purple'), dash='dash')
            ))
            
            # Add prediction intervals if available
            if lower_col in forecast_df.columns and upper_col in forecast_df.columns:
                fig.add_trace(go.Scatter(
                    x=filtered_forecast['Date'],
                    y=filtered_forecast[upper_col],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=filtered_forecast['Date'],
                    y=filtered_forecast[lower_col],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor=f"rgba({','.join(map(str, px.colors.hex_to_rgb(colors.get(model_name, '#9370DB'))))},0.2)",
                    name=f'{model_name} Prediction Interval',
                ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price Forecast",
        xaxis_title="Date",
        yaxis_title="Stock Price (IDR)",
        legend_title="Models",
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig

def create_performance_plot(metrics):
    """Create a bar plot comparing model performance metrics"""
    if not metrics:
        return go.Figure().update_layout(title="No metrics data available")
    
    # Extract models and metrics
    models = list(metrics.keys())
    metric_names = ['MAE', 'RMSE', 'MAPE', 'R2']
    
    # Create subplots
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=("Mean Absolute Error (MAE)", 
                                        "Root Mean Squared Error (RMSE)", 
                                        "Mean Absolute Percentage Error (MAPE)", 
                                        "R² Score"),
                        vertical_spacing=0.15)
    
    # Colors for different models
    colors = {'ARIMA': 'blue', 'Prophet': 'red', 'Ensemble': 'green'}
    
    # Add bar charts for each metric
    for i, metric in enumerate(metric_names):
        row = i // 2 + 1
        col = i % 2 + 1
        
        values = []
        model_names = []
        bar_colors = []
        
        for model in models:
            if metric in metrics[model].columns:
                values.append(metrics[model][metric].values[0])
                model_names.append(model)
                bar_colors.append(colors.get(model, 'purple'))
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                name=metric,
                marker_color=bar_colors
            ),
            row=row, col=col
        )
        
        # For MAPE and R2, lower is not always better
        yaxis_title = metric + (" (%)" if metric == "MAPE" else "")
        fig.update_yaxes(title_text=yaxis_title, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        height=700, 
        showlegend=False,
        title_text="Model Performance Comparison",
        template="plotly_white"
    )
    
    return fig

# App layout
app.layout = html.Div([
    html.H1("IDX Stock Price Forecasting Dashboard", 
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 40}),
    
    html.Div([
        html.Div([
            html.H3("Select Ticker:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
                value=TICKERS[0],
                clearable=False
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Select Timeframe:"),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[
                    {'label': '1 Month', 'value': '1M'},
                    {'label': '3 Months', 'value': '3M'},
                    {'label': '6 Months', 'value': '6M'},
                    {'label': '1 Year', 'value': '1Y'},
                    {'label': '3 Years', 'value': '3Y'},
                    {'label': '5 Years', 'value': '5Y'},
                    {'label': 'All Data', 'value': 'All'}
                ],
                value='3M',
                clearable=False
            ),
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("Select Models:"),
            dcc.Checklist(
                id='model-checklist',
                options=[
                    {'label': ' ARIMA', 'value': 'ARIMA'},
                    {'label': ' Prophet', 'value': 'Prophet'},
                    {'label': ' Ensemble', 'value': 'Ensemble'}
                ],
                value=['ARIMA', 'Prophet', 'Ensemble'],
                inline=True
            ),
        ], style={'width': '40%', 'display': 'inline-block'}),
    ]),
    
    # Forecast plot
    html.Div([
        dcc.Graph(id='forecast-plot')
    ]),
    
    # Performance metrics plot
    html.Div([
        html.H2("Model Performance Metrics", style={'textAlign': 'center'}),
        dcc.Graph(id='metrics-plot')
    ]),
    
    # Footer
    html.Div([
        html.P("IDX Stock Price Forecasting Dashboard - Developed with Plotly Dash", 
               style={'textAlign': 'center', 'color': 'gray', 'padding': '20px'})
    ])
])

# Callback for updating the forecast plot
@callback(
    Output('forecast-plot', 'figure'),
    [Input('ticker-dropdown', 'value'),
     Input('timeframe-dropdown', 'value'),
     Input('model-checklist', 'value')]
)
def update_forecast_plot(ticker, timeframe, selected_models):
    # Load the data
    actual_data, all_forecasts, _ = load_data(ticker)
    
    # Filter forecasts based on selected models
    selected_forecasts = {model: forecast for model, forecast in all_forecasts.items() 
                          if model in selected_models}
    
    # Create the plot
    fig = create_forecast_plot(ticker, actual_data, selected_forecasts, timeframe)
    return fig

# Callback for updating the metrics plot
@callback(
    Output('metrics-plot', 'figure'),
    [Input('ticker-dropdown', 'value'),
     Input('model-checklist', 'value')]
)
def update_metrics_plot(ticker, selected_models):
    # Load the data
    _, _, all_metrics = load_data(ticker)
    
    # Filter metrics based on selected models
    selected_metrics = {model: metrics for model, metrics in all_metrics.items() 
                          if model in selected_models}
    
    # Create the plot
    fig = create_performance_plot(selected_metrics)
    return fig

def run_dashboard_server():
    """Run the Dash server"""
    app.run_server(debug=True, port=8050)

if __name__ == '__main__':
    run_dashboard_server()
