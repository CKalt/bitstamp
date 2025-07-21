#!/bin/bash
# Deploy and run the SHORT position fix on server

echo "Deploying SHORT position fix to server..."

# Copy fix script to server
scp fix_short_position_server.py chris@your-server:/home/chris/projects/bitstamp/

# Execute fix on server
ssh chris@your-server << 'EOF'
cd /home/chris/projects/bitstamp
echo "Running fix script..."
python3 fix_short_position_server.py
EOF

echo "Fix deployment complete!"