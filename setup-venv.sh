#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display messages
function echo_info {
    echo -e "\033[1;34m$1\033[0m" # Blue colored text
}

function echo_success {
    echo -e "\033[1;32m$1\033[0m" # Green colored text
}

function echo_error {
    echo -e "\033[1;31m$1\033[0m" # Red colored text
}

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo_error "requirements.txt not found. Please ensure it's in the same directory as this script."
    exit 1
fi

# Create a new virtual environment
echo_info "Creating a new virtual environment..."
python3 -m venv env

# Activate the virtual environment
echo_info "Activating the virtual environment..."
source env/bin/activate || { echo_error "Failed to activate virtual environment"; exit 1; }

# Upgrade pip to the latest version
echo_info "Upgrading pip..."
pip install -U pip

# Install packages from requirements.txt
echo_info "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Print installed packages for verification
echo_info "Installed Python packages:"
pip list

echo_success "Virtual environment setup complete!"

# Deactivate the virtual environment
echo_info "Deactivating virtual environment..."
deactivate

echo_success "Script completed successfully!"