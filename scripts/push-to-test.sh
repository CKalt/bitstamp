#\!/bin/bash

cd /Users/chris/projects/python/btc-testing || exit 1

# First ensure we've pushed to origin
echo "=== Pushing to origin ==="
git push -u origin development

# Then pull on server
echo -e "\n=== Pulling on server ==="
ssh ck "cd /home/chris/projects/bitstamp-testing && git pull origin development"

echo -e "\n=== Done ==="
