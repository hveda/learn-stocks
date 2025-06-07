# IDX Stock Market Analysis Project

## Overview
This project analyzes Indonesian Stock Exchange (IDX) data using statistical models and machine learning techniques. The main focus is on BBCA.JK (Bank Central Asia) ticker data, but the infrastructure is designed to accommodate multiple stocks.

## Project Structure
```
machine-learning/
├── config/               # Configuration files and settings
│   └── settings.py       # Centralized configuration parameters
├── data/                 # Data storage
│   ├── raw/              # Raw data files from Yahoo Finance
│   ├── cleaned/          # Cleaned and preprocessed data 
│   ├── normalized/       # Normalized data for model training
│   ├── indicators/       # Technical indicators data
│   ├── features/         # Feature-engineered data
│   └── split/            # Train/test split data
├── docs/                 # Documentation files
├── logs/                 # Log files from various processes
├── models/               # Trained models
│   ├── arima/            # ARIMA models
│   ├── prophet/          # Prophet models
│   └── ensemble/         # Ensemble models
├── notebooks/            # Jupyter notebooks for exploration
├── results/              # Results from model runs
│   ├── arima/            # ARIMA model results and forecasts
│   ├── prophet/          # Prophet model results and forecasts
│   ├── ensemble/         # Ensemble model results and forecasts
│   ├── eda/              # Exploratory data analysis results
│   │   └── plots/        # EDA visualizations
│   ├── reports/          # Generated reports
│   ├── risk_assessment/  # Risk assessment outputs
│   └── alerts/           # Market alerts
├── scripts/              # Utility scripts
│   ├── check_metrics.py  # Script to check model metrics
│   ├── dashboard.py      # Interactive dashboard
│   ├── debug_summary.py  # Debug summary utility
│   ├── install_eda_packages.sh # Package installation script
│   ├── model_analysis.py # Model analysis runner
│   ├── run_dashboard.py  # Dashboard launcher
│   ├── run_ensemble_pipeline.sh # Shell script for ensemble pipeline
│   ├── run_model_analysis.py # Model analysis script
│   ├── run_pipeline.sh   # Shell script for full pipeline
│   ├── setup_project.py  # Project setup script
│   ├── test_ensemble.py  # Ensemble model test
│   └── update_technical_summary.py # Technical summary updater
├── src/                  # Source code
│   ├── data_collection/  # Data collection modules
│   ├── data_processing/  # Data processing modules
│   ├── exploration/      # Exploratory data analysis
│   ├── analysis/         # Analysis modules
│   ├── models/           # Model implementation
│   └── visualization/    # Visualization modules incl. dashboard
└── tests/                # Test suite
```

## Installation
1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Install visualization packages (optional):
   ```
   ./scripts/install_eda_packages.sh
   ```

## Setup
Run the setup script to create the necessary directories and initialize the project:
```
python run.py --setup
```

## Usage
The `run.py` script provides a unified interface for running different parts of the pipeline:

```
python run.py --collect     # Collect data from Yahoo Finance
python run.py --process     # Process raw data
python run.py --explore     # Run exploratory data analysis
python run.py --train       # Train all models
python run.py --train-arima # Train only ARIMA model
python run.py --train-prophet # Train only Prophet model
python run.py --train-ensemble # Train only Ensemble model
python run.py --report      # Generate performance reports
python run.py --risk        # Run risk assessment
python run.py --alerts      # Check market alerts
python run.py --all         # Run complete pipeline
```

You can also run the complete pipeline with the shell script:
```
./scripts/run_pipeline.sh
```

## Interactive Dashboard
Launch the interactive dashboard to visualize the results:
```
python scripts/run_dashboard.py
```
Then open your browser at http://127.0.0.1:8050/

## Models
The project implements three main forecasting approaches:
1. **ARIMA** (AutoRegressive Integrated Moving Average)
2. **Prophet** (Facebook/Meta's time series forecasting tool)
3. **Ensemble** (Combining predictions from multiple models)

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.
