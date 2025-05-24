#!/bin/zsh
# Script to run the full ML pipeline

echo "Creating necessary directories..."
mkdir -p data/raw
mkdir -p data/cleaned
mkdir -p data/normalized
mkdir -p data/indicators
mkdir -p data/features
mkdir -p data/split

echo "Running data collection..."
python run.py --collect

echo "Running data processing..."
python run.py --process

echo "Pipeline completed!"
echo "Checking output directories:"
echo "Raw data files:"
ls -la data/raw/
echo "Cleaned data files:"
ls -la data/cleaned/
echo "Normalized data files:"
ls -la data/normalized/
echo "Technical indicators files:"
ls -la data/indicators/
echo "Feature engineering files:"
ls -la data/features/
echo "Split data files:"
ls -la data/split/
