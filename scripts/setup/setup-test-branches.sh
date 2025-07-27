#!/bin/bash
# Setup test branches from current stable branch

source ~/ggmap

echo "Setting up test branches from stable-added-adaptive-trad-n-chart-more..."
echo "======================================================================="

# Setup on Mac
echo -e "\n1. Setting up Mac test branch..."
cd /Users/chris/projects/python/btc-testing
git fetch origin
# Delete existing development branch if it exists
git branch -D development 2>/dev/null
git checkout -b development origin/stable-added-adaptive-trad-n-chart-more
echo "✅ Mac test branch created from stable"

# Server already has development branch that was merged with stable
echo -e "\n2. Server test branch status..."
ssh ck "cd /home/chris/projects/bitstamp-testing && git branch --show-current && git log --oneline -1"
echo "✅ Server already on development branch with latest stable merged"

# Show status
echo -e "\n=== BRANCH STATUS ==="
echo "Mac (gg btc):    $(cd /Users/chris/projects/python/btc && git branch --show-current)"
echo "Mac (gg tst):    $(cd /Users/chris/projects/python/btc-testing && git branch --show-current)"
echo "Server (gg btc): $(ssh ck 'cd /home/chris/projects/bitstamp && git branch --show-current')"
echo "Server (gg tst): $(ssh ck 'cd /home/chris/projects/bitstamp-testing && git branch --show-current')"

echo -e "\n✅ Both test environments now branched from stable!"
echo "Ready for safe development and testing with 0.001 BTC positions."