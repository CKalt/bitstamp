#!/bin/sh
# Create virtual environment using Python 3.11
python3.11 -m venv env

# Activate the virtual environment
source env/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install numpy first to satisfy statsmodels build requirements
pip install numpy==1.24.3

# Install the rest of the requirements
pip install -r requirements.txt

