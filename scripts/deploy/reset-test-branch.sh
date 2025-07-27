#!/bin/bash
# Reset test branch to match current stable

source ~/ggmap

echo "Resetting test branches to stable-added-adaptive-trad-n-chart-more..."
echo "===================================================================="

# Reset Mac test branch
echo -e "\n1. Resetting Mac test branch..."
cd /Users/chris/projects/python/btc-testing
git fetch origin
git checkout -B development origin/stable-added-adaptive-trad-n-chart-more
echo "Latest commit: $(git log --oneline -1)"

echo -e "\n=== CURRENT STATUS ==="
echo "Mac (gg btc):    $(cd /Users/chris/projects/python/btc && git branch --show-current)"
echo "Mac (gg tst):    $(cd /Users/chris/projects/python/btc-testing && git branch --show-current)"
echo "Server (gg btc): $(ssh ck 'cd /home/chris/projects/bitstamp && git branch --show-current')"
echo "Server (gg tst): $(ssh ck 'cd /home/chris/projects/bitstamp-testing && git branch --show-current')"

echo -e "\n✅ Both test environments now have fresh 'development' branches from stable!"