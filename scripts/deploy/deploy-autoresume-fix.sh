#!/bin/bash
# Deploy auto-resume fix to test environment

echo "Deploying auto-resume fix to test environment..."
echo "=============================================="

# Copy the fixed file to test directory
cp /Users/chris/projects/python/btc-testing/src/tdr_server.py /tmp/tdr_server_fixed.py

# Commit in test directory
cd /Users/chris/projects/python/btc-testing
git add src/tdr_server.py
git commit -m "Fix auto-resume forcing True - respect config setting

- Don't force auto_resume=True when resume_auto_trade command is sent  
- Check auto_resume config in auto_resume_trading function
- Return proper error message when auto_resume is disabled"

# Push to origin
git push origin development

# Deploy to server
echo -e "\nDeploying to test server..."
ssh ck "cd /home/chris/projects/bitstamp-testing && git pull origin development"

echo -e "\nRestarting test server to apply fix..."
ssh ck "screen -S server-tst -X quit"
sleep 2
ssh ck "cd /home/chris/projects/bitstamp-testing && ./start-dev.sh"

echo -e "\n✅ Auto-resume fix deployed to test environment!"
echo "Next: Test that auto_resume=false is now respected"