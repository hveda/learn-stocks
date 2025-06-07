#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for running ensemble analysis
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
script_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(script_dir))

# Import configuration
from config.settings import DEFAULT_TICKER

def main():
    """Run the ensemble model test"""
    from src.models.ensemble_model import run_ensemble_analysis
    
    logging.basicConfig(level=logging.INFO)
    
    # Run ensemble analysis
    run_ensemble_analysis(DEFAULT_TICKER, method='weighted_average')
    print(f"Ensemble analysis completed for {DEFAULT_TICKER}.")

if __name__ == "__main__":
    main()
