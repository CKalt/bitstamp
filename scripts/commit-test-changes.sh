#!/bin/bash

# Script to commit changes in the test directory
cd /Users/chris/projects/python/btc-testing || exit 1

echo "=== Git Status Before ===" 
git status

echo -e "\n=== Adding Changes ==="
git add -A

echo -e "\n=== Committing Changes ==="
git commit -m "$(cat <<'EOF'
Integrate System Verifier into trading strategies

- Add SystemVerifier initialization to MACrossoverStrategy
- Add SystemVerifier initialization to AdaptiveMACrossoverStrategy  
- Run verification checks every 30 seconds in both strategy loops
- Log any detected issues for early regression detection
- Gracefully handle initialization failures

This provides continuous runtime verification to catch:
- Position tracking inconsistencies
- Entry price calculation errors
- Balance integrity issues
- Cost basis accuracy problems
- Trade execution failures
- Data synchronization mismatches

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"

echo -e "\n=== Pushing to Origin ==="
git push origin development

echo -e "\n=== Final Status ==="
git status