#!/bin/zsh
# Script to run the full ML pipeline

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")
# Get the project root (parent directory of the scripts directory)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

echo "Creating necessary directories..."
mkdir -p $PROJECT_ROOT/data/raw
mkdir -p $PROJECT_ROOT/data/cleaned
mkdir -p $PROJECT_ROOT/data/normalized
mkdir -p $PROJECT_ROOT/data/indicators
mkdir -p $PROJECT_ROOT/data/features
mkdir -p $PROJECT_ROOT/data/split
mkdir -p $PROJECT_ROOT/results/arima
mkdir -p $PROJECT_ROOT/results/prophet
mkdir -p $PROJECT_ROOT/results/reports

echo "Running data collection..."
cd $PROJECT_ROOT && python run.py --collect

echo "Running data processing..."
cd $PROJECT_ROOT && python run.py --process

echo "Running exploratory analysis..."
cd $PROJECT_ROOT && python run.py --explore

echo "Training ARIMA model..."
cd $PROJECT_ROOT && python run.py --train-arima

echo "Training Prophet model..."
cd $PROJECT_ROOT && python run.py --train-prophet

echo "Generating reports..."
cd $PROJECT_ROOT && python run.py --report

echo "Updating technical summary..."
cd $PROJECT_ROOT && python scripts/update_technical_summary.py

echo "Pipeline completed successfully!"

# Optional: Generate visualizations for presentation
echo "Would you like to generate visualizations for presentation? (y/n)"
read response
if [[ $response == "y" || $response == "Y" ]]; then
    echo "Generating visualizations..."
    cd $PROJECT_ROOT && python run.py --visualize
    echo "Visualizations generated!"
fi
