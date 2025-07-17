#!/bin/bash
# Script to set up environment variables for Bitstamp API credentials
# This is more secure than storing in files

echo "This script will help you set up environment variables for API credentials"
echo "Your credentials will NOT be displayed on screen"
echo ""

# Check if .bitstamp exists
if [ -f ".bitstamp" ]; then
    echo "Found .bitstamp file. You can extract values from it manually."
    echo ""
fi

echo "Add these lines to your ~/.bashrc or ~/.profile:"
echo ""
echo "# Bitstamp API Credentials"
echo "export BITSTAMP_CUSTOMER_ID='your_customer_id_here'"
echo "export BITSTAMP_API_KEY='your_api_key_here'"
echo "export BITSTAMP_API_SECRET='your_api_secret_here'"
echo ""
echo "Then run: source ~/.bashrc"
echo ""
echo "For extra security, consider using a password manager or"
echo "encrypted storage for these environment variables."