#!/bin/zsh
# Script to install required packages for exploratory data analysis

# Install EDA packages
echo "Installing visualization packages..."
pip install matplotlib==3.8.3
pip install seaborn==0.13.2
pip install plotly==5.21.0
pip install mplfinance==0.12.10b0
pip install prophet==1.1.5  # Facebook Prophet for time series forecasting

# List installed packages
echo "\nInstalled packages:"
pip list | grep -E 'matplotlib|seaborn|plotly|mplfinance|prophet'

echo "\nInstallation complete!"
