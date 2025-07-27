#\!/bin/bash

# First check local status
echo "=== Local btc-testing status ==="
cd /Users/chris/projects/python/btc-testing
git status
git branch -r | grep development || echo "No remote development branch found"

# Push to ensure remote has development branch
echo -e "\n=== Pushing development branch ==="
git push -u origin development

# Now pull on server
echo -e "\n=== Server operations ==="
ssh ck "cd /home/chris/projects/bitstamp-testing && echo 'Current branch:' && git branch && echo -e '\nFetching...' && git fetch origin && echo -e '\nPulling development...' && git pull origin development"

echo -e "\n=== Verifying System Verifier file exists ==="
ssh ck "ls -la /home/chris/projects/bitstamp-testing/src/tdr_core/system_verifier.py"
