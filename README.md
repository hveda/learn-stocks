# IDX Stock Market Analysis and Prediction

This project implements a machine learning pipeline for stock market data analysis and prediction, focusing on Indonesian Stock Exchange (IDX) data. The project follows a structured pipeline approach with clear separation of concerns.

## Project Structure

```
machine-learning/
├── data/                      # Data directory
│   ├── raw/                   # Raw data collected from Yahoo Finance
│   ├── cleaned/               # Data after cleaning (handling missing values, etc.)
│   ├── normalized/            # Normalized and standardized data
│   ├── indicators/            # Data with technical indicators added
│   ├── features/              # Data with engineered features
│   └── split/                 # Train/validation/test splits
├── src/                       # Source code
│   ├── data_collection/       # Code for collecting data
│   │   └── collect_yahoo_data.py  # Yahoo Finance data collector
│   ├── data_processing/       # Code for processing data
│   │   ├── data_cleaner.py    # Data cleaning module
│   │   ├── feature_engineering.py  # Feature engineering and splitting module
│   │   ├── process_data.py    # Main data processing orchestrator
│   │   └── technical_indicators.py  # Technical indicators module
│   ├── exploratory_analysis/  # Code for exploratory data analysis
│   ├── modeling/              # Code for model development
│   └── reporting/             # Code for generating reports and visualizations
├── run.py                     # Main runner script with CLI interface
├── run_pipeline.sh            # Shell script to run the full pipeline
├── requirements.txt           # Python dependencies
└── todo.md                    # Project roadmap and to-do list
```

## Data Processing Pipeline

The data processing pipeline consists of the following steps:

1. **Data Collection**
   - Fetch historical stock data from Yahoo Finance API for IDX tickers
   - Store raw data in CSV files (OHLCV data and company info)

2. **Data Cleaning**
   - Handle missing values
   - Remove duplicates
   - Adjust timestamps
   - Basic data validation

3. **Normalization and Technical Indicators**
   - Apply Min-Max scaling and standardization
   - Calculate technical indicators (Moving Averages, RSI, MACD, etc.)
   - Store processed data in separate files

4. **Feature Engineering**
   - Create lagged features
   - Generate time-based features
   - Create target variables for prediction
   - Engineer cross-features from technical indicators

5. **Time Series Splitting**
   - Split data into training, validation, and test sets
   - Respect chronological order of time series data

## Key Features

- Data collection from Yahoo Finance API for IDX stocks
- Historical data analysis going back to 2000 (where available)
- Comprehensive exploratory data analysis with visualizations
- Time series forecasting using ARIMA and Prophet models
- Medium-term trend predictions (weeks to months)
- Performance evaluation and reporting

## Project Structure

```
machine-learning/
├── README.md                 # Project documentation
├── todo.md                   # Project task list
├── requirements.txt          # Python dependencies
├── data/                     # Data directory
│   ├── raw/                  # Raw data from Yahoo Finance
│   └── processed/            # Cleaned and processed data
├── notebooks/                # Jupyter notebooks for analysis
├── src/                      # Source code
│   ├── data_collection/      # Scripts for data collection
│   ├── data_processing/      # Data cleaning and preprocessing
│   ├── exploration/          # EDA scripts
│   ├── models/               # ARIMA and Prophet model implementations
│   └── visualization/        # Visualization and reporting scripts
└── results/                  # Model outputs and visualizations
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd /path/to/machine-learning
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Collection

To collect stock data from Yahoo Finance:

```bash
python src/data_collection/collect_data.py --start_date 2000-01-01 --tickers TLKM.JK BBCA.JK
```

### Exploratory Analysis

Run the exploratory analysis notebook:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

### Model Training

To train the ARIMA and Prophet models:

```bash
python src/models/train_arima.py --ticker TLKM.JK
python src/models/train_prophet.py --ticker TLKM.JK
```

### Generating Reports

To generate prediction reports and visualizations:

```bash
python src/visualization/generate_reports.py --ticker TLKM.JK --model arima
```

## Methodology

### Data Collection
Data is collected from Yahoo Finance API with rate limiting to respect API constraints. The focus is on IDX stocks with historical data going back to 2000 where available.

### Data Processing
The data processing workflow includes handling missing values, normalization, calculating technical indicators, and preparing data for time series analysis.

### Modeling
The project employs two primary forecasting models:

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Traditional statistical method for time series forecasting
   - Good for capturing linear relationships in data
   - Parameters are tuned using AIC/BIC criteria

2. **Prophet**
   - Developed by Facebook/Meta
   - Robust to missing data and outliers
   - Better at capturing seasonality and trend changes
   - Handles special events and holidays

Both models are configured to forecast medium-term (weeks to months) price trends.

## Visualization

The project uses two main visualization libraries:

1. **Matplotlib**
   - Used for static, publication-quality visualizations
   - Time series plotting, correlation matrices, etc.

2. **Plotly**
   - Used for interactive visualizations
   - Allows for zooming, hovering, and exploration of data
   - Used in dashboards for model insights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
