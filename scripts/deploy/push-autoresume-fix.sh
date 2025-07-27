#!/bin/bash
# Push auto-resume fix from Mac test directory

echo "Pushing auto-resume fix from Mac..."
echo "==================================="

cd /Users/chris/projects/python/btc-testing

# Add and commit
git add -A
git commit -m "Fix auto-resume forcing True - respect config setting

- Don't force auto_resume=True when resume_auto_trade command is sent
- Check auto_resume config in auto_resume_trading function  
- Return proper error message when auto_resume is disabled"

# Push to origin
git push origin development

echo -e "\nPulling changes on server..."
ssh ck "cd /home/chris/projects/bitstamp-testing && git pull origin development"

echo -e "\nRestarting test server..."
ssh ck "screen -S server-tst -X quit"
sleep 2
ssh ck "cd /home/chris/projects/bitstamp-testing && ./start-dev.sh"

echo -e "\n✅ Fix deployed!"