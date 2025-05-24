#!/usr/bin/env python3
"""
Test script for Yahoo Finance data collection.
"""

import unittest
from pathlib import Path
import pandas as pd
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_collection.yahoo_finance_crawler import (
    fetch_stock_data,
    validate_data,
    setup_logging
)


class TestYahooFinanceCrawler(unittest.TestCase):
    """Test cases for Yahoo Finance data crawler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = setup_logging()
        self.test_ticker = "BBCA.JK"
        self.start_date = "2023-01-01"
        self.end_date = "2023-01-31"
    
    def test_fetch_stock_data(self):
        """Test fetching stock data."""
        df = fetch_stock_data(
            self.test_ticker, 
            self.start_date, 
            self.end_date, 
            self.logger
        )
        
        # Check that data was fetched
        self.assertIsNotNone(df, "Failed to fetch data")
        
        # Check that we have the expected columns
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for column in expected_columns:
            self.assertIn(column, df.columns, f"Missing column: {column}")
        
        # Check that we have some data
        self.assertGreater(len(df), 0, "No data rows returned")
    
    def test_validate_data(self):
        """Test data validation."""
        # Create test data with some issues
        test_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, None, 104.0],
            'High': [105.0, 106.0, 107.0, 108.0, 109.0],
            'Low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'Close': [101.0, 102.0, 103.0, 104.0, 1040.0],  # Outlier in last value
            'Volume': [10000, 10100, 10200, 10300, 10400]
        }, index=pd.date_range(start='2023-01-01', periods=5))
        
        # Add a duplicate index
        duplicate_data = pd.DataFrame({
            'Open': [103.0],
            'High': [107.5],
            'Low': [97.5],
            'Close': [103.5],
            'Volume': [10250]
        }, index=pd.date_range(start='2023-01-03', periods=1))
        
        test_data = pd.concat([test_data, duplicate_data])
        
        # Validate data
        cleaned_df = validate_data(test_data, self.test_ticker, self.logger)
        
        # Check that duplicate was removed
        self.assertEqual(len(cleaned_df), 5, "Duplicate was not properly handled")
        
        # Missing values remain (this is expected - data cleaning happens separately)
        self.assertTrue(cleaned_df['Open'].isna().any(), "Missing values should still be present")


if __name__ == '__main__':
    unittest.main()
