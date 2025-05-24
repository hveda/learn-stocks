#!/usr/bin/env python3
"""
Interactive Dashboard for Stock Price Forecasting Models

This script creates an interactive dashboard using Plotly Dash
to visualize and compare the forecasting results from different models.
"""
import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dashboard.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Define paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR = PROJECT_ROOT / 'data'
REPORTS_DIR = RESULTS_DIR / 'reports'
ARIMA_RESULTS_DIR = RESULTS_DIR / 'arima'
PROPHET_RESULTS_DIR = RESULTS_DIR / 'prophet'

# List of tickers to display
TICKERS = ['BBCA.JK', 'PTBA.JK']

# Initialize the Dash app
app = dash.Dash(__name__, title="Stock Price Forecasting Dashboard")

# Function to load data
def load_data(ticker):
    """Load historical and forecast data for a ticker"""
    ticker_clean = ticker.replace('.', '_')
    data = {}
    
    try:
        # Load historical data
        cleaned_data_path = DATA_DIR / 'cleaned' / f"{ticker_clean}.csv"
        if cleaned_data_path.exists():
            data['historical'] = pd.read_csv(cleaned_data_path, parse_dates=['Date'])
            logger.info(f"Loaded historical data for {ticker}")
        
        # Load ARIMA forecasts
        arima_forecast_path = ARIMA_RESULTS_DIR / f"{ticker_clean}_future_forecast.csv"
        if arima_forecast_path.exists():
            data['arima_forecast'] = pd.read_csv(arima_forecast_path, parse_dates=[0])
            logger.info(f"Loaded ARIMA forecast for {ticker}")
        
        # Load Prophet forecasts
        prophet_forecast_path = PROPHET_RESULTS_DIR / f"{ticker_clean}_future_forecast.csv"
        if prophet_forecast_path.exists():
            data['prophet_forecast'] = pd.read_csv(prophet_forecast_path)
            # Convert ds to datetime if needed
            if 'ds' in data['prophet_forecast'].columns:
                data['prophet_forecast']['ds'] = pd.to_datetime(data['prophet_forecast']['ds'])
            logger.info(f"Loaded Prophet forecast for {ticker}")
        
        # Load model metrics
        model_metrics_path = REPORTS_DIR / f"{ticker_clean}_model_comparison.csv"
        if model_metrics_path.exists():
            data['metrics'] = pd.read_csv(model_metrics_path)
            logger.info(f"Loaded model metrics for {ticker}")
            
        return data
    
    except Exception as e:
        logger.error(f"Error loading data for {ticker}: {str(e)}")
        return {}

# Load data for all tickers
ticker_data = {ticker: load_data(ticker) for ticker in TICKERS}

# Create the app layout
app.layout = html.Div([
    html.H1("Stock Price Forecasting Dashboard", 
            style={'textAlign': 'center', 'marginBottom': 30, 'marginTop': 20}),
    
    html.Div([
        html.Div([
            html.H3("Select Options", style={'marginBottom': 15}),
            html.Label("Select Ticker:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': ticker, 'value': ticker} for ticker in TICKERS],
                value=TICKERS[0],
                clearable=False,
                style={'width': '100%', 'marginBottom': 15}
            ),
            
            html.Label("Select View:"),
            dcc.RadioItems(
                id='view-selector',
                options=[
                    {'label': 'Price Forecasts', 'value': 'forecasts'},
                    {'label': 'Model Performance', 'value': 'performance'},
                    {'label': 'Model Comparison', 'value': 'comparison'}
                ],
                value='forecasts',
                style={'marginBottom': 15}
            ),
            
            html.Label("Date Range:"),
            dcc.RangeSlider(
                id='date-range-slider',
                min=0,
                max=100,
                value=[80, 100],
                marks={0: 'Start', 50: 'Mid', 100: 'End'},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            
            html.Div(id='metrics-display', style={'marginTop': 30}),
        ], style={'width': '25%', 'float': 'left', 'padding': '20px', 'backgroundColor': '#f9f9f9', 'height': '100vh'}),
        
        html.Div([
            dcc.Graph(id='main-graph', style={'height': '70vh'}),
            dcc.Graph(id='secondary-graph', style={'height': '30vh'})
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
    ])
])

@callback(
    [Output('main-graph', 'figure'),
     Output('secondary-graph', 'figure'),
     Output('metrics-display', 'children'),
     Output('date-range-slider', 'marks')],
    [Input('ticker-dropdown', 'value'),
     Input('view-selector', 'value'),
     Input('date-range-slider', 'value')]
)
def update_graphs(selected_ticker, selected_view, date_range):
    """Update the graphs based on user selections"""
    data = ticker_data.get(selected_ticker, {})
    
    if not data:
        return go.Figure(), go.Figure(), html.P("No data available"), {}
    
    # Default marks for the slider
    marks = {0: 'Start', 50: 'Mid', 100: 'End'}
    
    # Create the metrics display
    metrics_display = []
    if 'metrics' in data:
        metrics_df = data['metrics']
        metrics_display = [
            html.H4("Model Metrics"),
            html.Table([
                html.Tr([html.Th("Model"), html.Th("MAE"), html.Th("RMSE"), html.Th("MAPE"), html.Th("R²")]),
                *[
                    html.Tr([
                        html.Td(row['Model']), 
                        html.Td(f"{row['MAE']:.4f}"), 
                        html.Td(f"{row['RMSE']:.4f}"), 
                        html.Td(f"{row['MAPE']:.2f}%"), 
                        html.Td(f"{row['R2']:.4f}")
                    ]) 
                    for _, row in metrics_df.iterrows()
                ]
            ], style={'width': '100%', 'border': '1px solid #ddd', 'borderCollapse': 'collapse'})
        ]
    
    if selected_view == 'forecasts':
        return create_forecast_view(data, selected_ticker, date_range), create_volume_view(data, selected_ticker, date_range), metrics_display, marks
    
    elif selected_view == 'performance':
        return create_performance_view(data, selected_ticker), create_error_distribution(data, selected_ticker), metrics_display, marks
    
    elif selected_view == 'comparison':
        return create_model_comparison(data, selected_ticker), create_forecast_comparison(data, selected_ticker), metrics_display, marks
    
    # Default case
    return go.Figure(), go.Figure(), metrics_display, marks

def create_forecast_view(data, ticker, date_range):
    """Create the main forecast view with historical and predicted prices"""
    fig = go.Figure()
    
    # Add historical data if available
    if 'historical' in data:
        hist_data = data['historical']
        # Calculate the date range based on slider
        total_days = (hist_data['Date'].max() - hist_data['Date'].min()).days
        start_day = hist_data['Date'].min() + pd.Timedelta(days=int(date_range[0]/100 * total_days))
        end_day = hist_data['Date'].min() + pd.Timedelta(days=int(date_range[1]/100 * total_days))
        
        # Filter by date range
        filtered_hist = hist_data[(hist_data['Date'] >= start_day) & (hist_data['Date'] <= end_day)]
        
        # Add candlestick chart
        fig.add_trace(go.Candlestick(
            x=filtered_hist['Date'],
            open=filtered_hist['Open'],
            high=filtered_hist['High'],
            low=filtered_hist['Low'],
            close=filtered_hist['Close'],
            name='Historical Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ))
    
    # Add ARIMA forecast if available
    if 'arima_forecast' in data:
        arima_data = data['arima_forecast']
        fig.add_trace(go.Scatter(
            x=arima_data.iloc[:, 0],
            y=arima_data['predicted_Close'] if 'predicted_Close' in arima_data.columns else arima_data['forecast'],
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='blue')
        ))
        
        # Add confidence intervals if available
        if 'lower_Close' in arima_data.columns and 'upper_Close' in arima_data.columns:
            fig.add_trace(go.Scatter(
                x=arima_data.iloc[:, 0],
                y=arima_data['upper_Close'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=arima_data.iloc[:, 0],
                y=arima_data['lower_Close'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0, 0, 255, 0.2)',
                name='ARIMA Confidence Interval'
            ))
    
    # Add Prophet forecast if available
    if 'prophet_forecast' in data:
        prophet_data = data['prophet_forecast']
        fig.add_trace(go.Scatter(
            x=prophet_data['ds'],
            y=prophet_data['yhat'],
            mode='lines',
            name='Prophet Forecast',
            line=dict(color='red')
        ))
        
        # Add confidence intervals if available
        if 'yhat_lower' in prophet_data.columns and 'yhat_upper' in prophet_data.columns:
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'],
                y=prophet_data['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=prophet_data['ds'],
                y=prophet_data['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(255, 0, 0, 0.2)',
                name='Prophet Confidence Interval'
            ))
    
    # Update the layout
    fig.update_layout(
        title=f"{ticker} Price Forecast",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def create_volume_view(data, ticker, date_range):
    """Create the volume chart as secondary view"""
    fig = go.Figure()
    
    if 'historical' in data:
        hist_data = data['historical']
        # Calculate the date range based on slider
        total_days = (hist_data['Date'].max() - hist_data['Date'].min()).days
        start_day = hist_data['Date'].min() + pd.Timedelta(days=int(date_range[0]/100 * total_days))
        end_day = hist_data['Date'].min() + pd.Timedelta(days=int(date_range[1]/100 * total_days))
        
        # Filter by date range
        filtered_hist = hist_data[(hist_data['Date'] >= start_day) & (hist_data['Date'] <= end_day)]
        
        # Add volume bars
        colors = ['red' if close < open else 'green' for close, open in zip(filtered_hist['Close'], filtered_hist['Open'])]
        
        fig.add_trace(go.Bar(
            x=filtered_hist['Date'],
            y=filtered_hist['Volume'],
            name='Volume',
            marker_color=colors
        ))
        
        # Add 20-day moving average of volume
        if len(filtered_hist) > 20:
            filtered_hist['Volume_MA20'] = filtered_hist['Volume'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(
                x=filtered_hist['Date'],
                y=filtered_hist['Volume_MA20'],
                name='20-Day Volume MA',
                line=dict(color='blue', width=2)
            ))
    
    # Update the layout
    fig.update_layout(
        title=f"{ticker} Trading Volume",
        xaxis_title="Date",
        yaxis_title="Volume",
        hovermode="x unified",
        showlegend=True,
        height=300
    )
    
    return fig

def create_performance_view(data, ticker):
    """Create a view to analyze model performance"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("ARIMA Residuals", "Prophet Residuals", 
                        "ARIMA Forecast vs Actual", "Prophet Forecast vs Actual")
    )
    
    # This is a placeholder for actual implementation
    # In a real scenario, we would load the model evaluation results and plot:
    # 1. Residual analysis (top row)
    # 2. Forecast vs Actual comparison (bottom row)
    
    # Add sample residual plots (would be replaced with actual data)
    x = np.arange(0, 100)
    arima_residuals = np.random.normal(0, 10, 100)  # Sample residuals
    prophet_residuals = np.random.normal(0, 15, 100)  # Sample residuals
    
    fig.add_trace(go.Scatter(x=x, y=arima_residuals, mode='markers', name='ARIMA Residuals'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=prophet_residuals, mode='markers', name='Prophet Residuals'), row=1, col=2)
    
    # Add reference line at y=0 for residual plots
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=0, line=dict(color="red", dash="dot"), row=1, col=1)
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=0, line=dict(color="red", dash="dot"), row=1, col=2)
    
    # Add sample forecast vs actual plots
    actual = np.cumsum(np.random.normal(0.5, 1, 100)) + 100  # Sample price data
    arima_forecast = actual + np.random.normal(0, 5, 100)  # Sample forecast
    prophet_forecast = actual + np.random.normal(0, 7, 100)  # Sample forecast
    
    fig.add_trace(go.Scatter(x=x, y=actual, mode='lines', name='Actual', line=dict(color='black')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=arima_forecast, mode='lines', name='ARIMA Forecast', line=dict(color='blue')), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=x, y=actual, mode='lines', name='Actual', line=dict(color='black'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=x, y=prophet_forecast, mode='lines', name='Prophet Forecast', line=dict(color='red')), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Model Performance Analysis",
        height=600,
        showlegend=True
    )
    
    # Update y-axis titles
    fig.update_yaxes(title_text="Residual", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=1, col=2)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=2)
    
    # Update x-axis titles
    fig.update_xaxes(title_text="Observation", row=1, col=1)
    fig.update_xaxes(title_text="Observation", row=1, col=2)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    
    return fig

def create_error_distribution(data, ticker):
    """Create error distribution plots"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ARIMA Error Distribution", "Prophet Error Distribution"))
    
    # Sample error distributions - would be replaced with actual data
    arima_errors = np.random.normal(0, 10, 1000)  # Sample errors
    prophet_errors = np.random.normal(0, 15, 1000)  # Sample errors
    
    fig.add_trace(go.Histogram(x=arima_errors, name='ARIMA Errors', opacity=0.7, marker_color='blue'), row=1, col=1)
    fig.add_trace(go.Histogram(x=prophet_errors, name='Prophet Errors', opacity=0.7, marker_color='red'), row=1, col=2)
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Error Distributions",
        height=300,
        showlegend=True
    )
    
    # Update axis titles
    fig.update_xaxes(title_text="Error", row=1, col=1)
    fig.update_xaxes(title_text="Error", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    return fig

def create_model_comparison(data, ticker):
    """Create bar charts comparing model performance metrics"""
    if 'metrics' not in data:
        fig = go.Figure()
        fig.add_annotation(text="No metrics data available", showarrow=False)
        return fig
    
    metrics_df = data['metrics']
    
    # Create a grouped bar chart for metrics comparison
    fig = go.Figure()
    
    models = metrics_df['Model'].tolist()
    
    # Add bars for each metric
    for metric in ['MAE', 'RMSE', 'MAPE']:
        if metric in metrics_df.columns:
            fig.add_trace(go.Bar(
                x=models,
                y=metrics_df[metric],
                name=metric
            ))
    
    # Add R² as a separate trace with different scale
    if 'R2' in metrics_df.columns:
        fig.add_trace(go.Scatter(
            x=models,
            y=metrics_df['R2'],
            mode='markers',
            marker=dict(size=15, symbol='star'),
            name='R²',
            yaxis='y2'
        ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Model Performance Comparison",
        xaxis=dict(title='Model'),
        yaxis=dict(title='Error Metrics (lower is better)'),
        yaxis2=dict(
            title='R² (higher is better)',
            overlaying='y',
            side='right',
            rangemode='tozero',
            range=[0, 1]
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group'
    )
    
    return fig

def create_forecast_comparison(data, ticker):
    """Create a plot comparing the forecast accuracy of different models"""
    fig = go.Figure()
    
    # In a real implementation, this would show the comparison of forecast accuracy
    # across different market conditions or time periods
    
    # Sample data for illustration
    conditions = ['Bullish', 'Bearish', 'Sideways', 'Volatile']
    arima_errors = [5.2, 8.7, 3.9, 10.5]  # Sample errors by market condition
    prophet_errors = [6.1, 7.5, 4.2, 9.8]  # Sample errors by market condition
    
    fig.add_trace(go.Bar(
        x=conditions,
        y=arima_errors,
        name='ARIMA Error',
        marker_color='blue'
    ))
    
    fig.add_trace(go.Bar(
        x=conditions,
        y=prophet_errors,
        name='Prophet Error',
        marker_color='red'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Forecast Errors by Market Condition",
        xaxis=dict(title='Market Condition'),
        yaxis=dict(title='Error (MAPE %)'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        barmode='group',
        height=300
    )
    
    return fig

# Run the server
if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
