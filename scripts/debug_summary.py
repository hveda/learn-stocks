#!/usr/bin/env python3
"""
Debug script to check why the technical summary update isn't working
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

from scripts import update_technical_summary as uts
from config.settings import PROJECT_ROOT

# Read metrics
metrics = uts.read_model_metrics()
print(f"Available metrics: {list(metrics.keys())}")

# Check if we have ensemble data
has_ensemble = any('ensemble' in k.lower() for k in metrics.keys())
print(f"Has ensemble data: {has_ensemble}")

# Check if the technical summary mentions combined models
summary_path = PROJECT_ROOT / 'CONSOLIDATED_TECHNICAL_SUMMARY.md'
with open(summary_path, 'r') as f:
    content = f.read()
has_combined = 'Combined Models' in content
print(f"Summary mentions Combined Models: {has_combined}")

# Check if the section marker exists
has_section = '## 8. What We Learned and Recommendations' in content
print(f"Has section marker: {has_section}")

# Generate verification note
verification_note = """

> **Note about the data**: This summary has been verified against the actual model outputs as of June 7, 2025.

"""

# Check if verification note exists or needs to be added
if verification_note.strip() not in content:
    print("Verification note is missing, should be added.")
else:
    print("Verification note exists.")

# Check file paths for metrics
print("\nChecking file paths:")
from config.settings import ARIMA_RESULTS_DIR, PROPHET_RESULTS_DIR, ENSEMBLE_RESULTS_DIR, TICKER_CLEAN

arima_metrics = ARIMA_RESULTS_DIR / f"{TICKER_CLEAN}_metrics.csv"
prophet_metrics = PROPHET_RESULTS_DIR / f"{TICKER_CLEAN}_metrics.csv" 
ensemble_metrics = ENSEMBLE_RESULTS_DIR / f"{TICKER_CLEAN}_metrics.csv"

print(f"ARIMA metrics path exists: {arima_metrics.exists()}")
print(f"Prophet metrics path exists: {prophet_metrics.exists()}")
print(f"Ensemble metrics path exists: {ensemble_metrics.exists()}")

if __name__ == "__main__":
    print("Debug summary completed.")
