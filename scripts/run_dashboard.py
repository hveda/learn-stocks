#!/usr/bin/env python3
"""
Script to launch the interactive dashboard for the IDX Stock Analysis project
"""
import os
import sys
from pathlib import Path

# Add the project root to Python path
script_path = Path(__file__).resolve()
project_root = script_path.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.dashboard import run_dashboard_server

if __name__ == "__main__":
    print("Starting the IDX Stock Analysis Dashboard...")
    print("Open http://127.0.0.1:8050/ in your browser to view the dashboard")
    run_dashboard_server()
