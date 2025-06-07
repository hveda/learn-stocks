#!/bin/bash
# This script runs the full pipeline including ensemble model implementation

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")
# Get the project root (parent directory of the scripts directory)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

# Set up proper environment
echo "Setting up environment..."
source $PROJECT_ROOT/venv/bin/activate || { echo "Virtual environment not found. Please create and activate it first."; exit 1; }

# Run data collection if needed
if [ ! -d "$PROJECT_ROOT/data/raw" ] || [ "$(ls -A $PROJECT_ROOT/data/raw 2>/dev/null)" == "" ]; then
    echo "Running data collection..."
    cd $PROJECT_ROOT && python run.py --collect
fi

# Run data processing
echo "Processing data..."
cd $PROJECT_ROOT && python run.py --process

# Run model training
echo "Training models (ARIMA, Prophet, and Ensemble)..."
cd $PROJECT_ROOT && python run.py --train

# Run ensemble model specifically
echo "Training ensemble model..."
cd $PROJECT_ROOT && python run.py --train-ensemble

# Generate reports
echo "Generating reports with ensemble results..."
cd $PROJECT_ROOT && python run.py --report

# Update technical summary
echo "Updating technical summary..."
cd $PROJECT_ROOT && python scripts/update_technical_summary.py

echo "Ensemble pipeline completed successfully!"

# Optional: Run risk assessment
echo "Running risk assessment..."
cd $PROJECT_ROOT && python run.py --risk
