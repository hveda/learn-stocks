#!/usr/bin/env python3
"""
Initialize the project structure and setup configuration files.
"""

import os
import logging
import sys
from pathlib import Path


def setup_logging():
    """
    Configure logging for the project.
    """
    logging_dir = Path("logs")
    logging_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logging_dir / "setup.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("project_setup")


def create_directory_structure(logger):
    """
    Create the directory structure for the project if it doesn't exist.
    """
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data_collection",
        "src/data_processing",
        "src/exploration",
        "src/models",
        "src/visualization",
        "results",
        "logs"
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        else:
            logger.info(f"Directory already exists: {path}")


def create_init_files(logger):
    """
    Create __init__.py files in all Python module directories.
    """
    python_dirs = [
        "src",
        "src/data_collection",
        "src/data_processing",
        "src/exploration",
        "src/models",
        "src/visualization"
    ]
    
    for directory in python_dirs:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            logger.info(f"Created {init_file}")
        else:
            logger.info(f"{init_file} already exists")


def create_env_file(logger):
    """
    Create a .env file for environment variables.
    """
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write("""# Environment variables for Stock Market ML Project
# Add any API keys or configuration settings here

# Yahoo Finance API configuration
# No API key needed for basic usage, but set rate limits
YAHOO_FINANCE_RATE_LIMIT=2000  # requests per hour

# Project paths
DATA_DIR=data
RESULTS_DIR=results
""")
        logger.info("Created .env file with default configurations")
    else:
        logger.info(".env file already exists")


def create_gitignore(logger):
    """
    Create a .gitignore file with common Python patterns.
    """
    gitignore_file = Path(".gitignore")
    if not gitignore_file.exists():
        with open(gitignore_file, "w") as f:
            f.write("""# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# Environment variables
.env

# Log files
logs/
*.log

# Data files (optionally track these with Git LFS)
# Uncomment if you don't want to track data files
# data/

# Results
results/

# IDE specific files
.idea/
.vscode/
*.swp
*.swo
""")
        logger.info("Created .gitignore file")
    else:
        logger.info(".gitignore file already exists")


def create_readme_stub(logger):
    """
    Ensure README.md exists with project title.
    """
    readme_file = Path("README.md")
    if not readme_file.exists():
        with open(readme_file, "w") as f:
            f.write("# IDX Stock Market Analysis and Prediction\n\n")
            f.write("See README.md for more information about this project.\n")
        logger.info("Created README.md stub")
    else:
        logger.info("README.md already exists")


def create_test_directory(logger):
    """
    Create a tests directory structure.
    """
    test_dirs = [
        "tests",
        "tests/data_collection",
        "tests/data_processing",
        "tests/models"
    ]
    
    for directory in test_dirs:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created test directory: {path}")
        else:
            logger.info(f"Test directory already exists: {path}")
    
    # Create test __init__.py files
    for directory in test_dirs:
        init_file = Path(directory) / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            logger.info(f"Created {init_file}")


def main():
    """
    Main function to initialize the project.
    """
    logger = setup_logging()
    logger.info("Starting project initialization")
    
    create_directory_structure(logger)
    create_init_files(logger)
    create_env_file(logger)
    create_gitignore(logger)
    create_readme_stub(logger)
    create_test_directory(logger)
    
    logger.info("Project initialization completed")
    print("\nProject structure initialized successfully!")
    print("Next steps:")
    print("1. Activate the virtual environment: source venv/bin/activate")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Start working on data collection for BBCA.JK and PTBA.JK")


if __name__ == "__main__":
    main()
