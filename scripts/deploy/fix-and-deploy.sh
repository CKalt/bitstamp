#!/bin/bash
# Properly commit and deploy the auto-resume fix

echo "Fixing auto-resume deployment..."
echo "================================"

cd /Users/chris/projects/python/btc-testing

# Check git status
echo "Git status:"
git status -s

# Stage the actual fix file
git add src/tdr_server.py

# Commit with the fix
git commit -m "Fix auto-resume: respect config setting instead of forcing True

- Check auto_resume config in auto_resume_trading function
- Don't force auto_resume=True in API endpoint
- Log when auto-resume is disabled by config"

# Push to development branch
git push origin development

# Pull on server
echo -e "\nDeploying to server..."
ssh ck "cd /home/chris/projects/bitstamp-testing && git pull origin development"

# Verify the fix is there
echo -e "\nVerifying fix on server..."
ssh ck "cd /home/chris/projects/bitstamp-testing && grep -n 'Auto-resume is disabled in configuration' src/tdr_server.py"

# Restart server
echo -e "\nRestarting server..."
ssh ck "screen -S server-tst -X quit"
sleep 2
ssh ck "cd /home/chris/projects/bitstamp-testing && ./start-dev.sh"

echo -e "\n✅ Done!"