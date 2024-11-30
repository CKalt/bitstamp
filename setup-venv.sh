#!/bin/sh
python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip

# Install numpy first to satisfy statsmodels build requirements
pip install numpy==1.24.3

# Install the rest of the requirements
pip install -r requirements.txt

