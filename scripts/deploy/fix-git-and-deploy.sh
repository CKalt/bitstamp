#!/bin/bash
# Fix git setup and deploy

echo "Fixing git and deploying..."
echo "=========================="

cd /Users/chris/projects/python/btc-testing

# First, push the development branch to origin
echo "Creating development branch on origin..."
git push -u origin development

# Now deploy to server
echo -e "\nDeploying to server..."
ssh ck "cd /home/chris/projects/bitstamp-testing && git fetch origin && git checkout development && git pull origin development"

# Verify
echo -e "\nVerifying fix is deployed..."
ssh ck "cd /home/chris/projects/bitstamp-testing && grep -n 'Auto-resume is disabled in configuration' src/tdr_server.py"

echo -e "\n✅ Should be fixed now!"