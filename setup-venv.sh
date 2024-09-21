#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to display messages
function echo_info {
    echo -e "\033[1;34m$1\033[0m"  # Blue colored text
}

function echo_success {
    echo -e "\033[1;32m$1\033[0m"  # Green colored text
}

function echo_error {
    echo -e "\033[1;31m$1\033[0m"  # Red colored text
}

# Create a new virtual environment
echo_info "Creating a new virtual environment..."
python3 -m venv env

# Activate the virtual environment
echo_info "Activating the virtual environment..."
source env/bin/activate

# Upgrade pip to the latest version
echo_info "Upgrading pip..."
pip install --upgrade pip

# Install compatible versions of required packages
echo_info "Installing required Python packages..."

# Upgrade to websockets 10.x to ensure compatibility
pip install websockets==10.4

# Install other dependencies with specified versions
pip install requests==2.27.1
pip install urllib3==1.26.15
pip install pandas==1.3.5
pip install psycopg2-binary==2.9.3
pip install websocket-client==1.3.3
pip install statsmodels==0.13.2
pip install httpx==0.23.0
pip install matplotlib==3.5.2  # Added as per your script

# Optional: Install any other required packages
# Uncomment and modify the line below if you need to install additional packages
# pip install package-name==version

# Print installed packages for verification
echo_info "Installed Python packages:"
pip list

echo_success "Virtual environment setup complete!"
