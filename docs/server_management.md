# TDR Server Management Guide

## Quick Reference

### Basic Commands
```bash
# Check server status
bin/server_control.sh status

# View live logs
bin/server_control.sh logs

# Restart server
bin/server_control.sh restart

# Quick restart with git pull
bin/quick_restart.sh
```

## Server Control Script

The `server_control.sh` script provides easy server management:

### Commands

| Command | Description |
|---------|-------------|
| `status` | Check if server is running and show details |
| `start` | Start the server (if not running) |
| `stop` | Stop the server gracefully |
| `restart` | Stop and start the server |
| `logs` | Show live server logs (tail -f) |
| `attach` | Attach to server output |
| `quick-restart` | Stop, git pull, and restart |

### Examples

```bash
# Check server status
$ bin/server_control.sh status
🔍 Checking server status...
✅ Server is running (PID: 12345)

📊 Server Details:
  "auto_trader": { "active": true }
  "history_loaded": true
  "websocket": "connected"

# Watch logs
$ bin/server_control.sh logs
📋 Showing server logs (Ctrl+C to exit)...
[live log output...]

# Restart server
$ bin/server_control.sh restart
🛑 Stopping server...
✅ Server stopped gracefully
🚀 Starting server...
✅ Server started successfully (PID: 12346)
```

## Process Management

### Finding the Server Process
```bash
# Find server PID
pgrep -f "python.*tdr_server.py"

# See full process details
ps aux | grep tdr_server
```

### Manual Process Control
```bash
# Kill server (if script fails)
pkill -f "python.*tdr_server.py"

# Start manually in foreground
python src/tdr_server.py

# Start manually in background
nohup python src/tdr_server.py > logs/server.log 2>&1 &
```

## Log Files

Server logs are stored in:
- `logs/server.log` - Main server output
- `logs/server_restart.log` - Quick restart script output
- `logs/tdr_server.log` - Detailed application logs

### Viewing Logs
```bash
# Last 100 lines
tail -100 logs/server.log

# Follow logs in real-time
tail -f logs/server.log

# Search logs
grep ERROR logs/server.log
grep "Pivot" logs/tdr_server.log
```

## Systemd Service (Optional)

For production deployments, use systemd:

### Install Service
```bash
# Copy service file
sudo cp tdr-server.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable auto-start
sudo systemctl enable tdr-server

# Start service
sudo systemctl start tdr-server
```

### Systemd Commands
```bash
# Status
sudo systemctl status tdr-server

# Start/Stop/Restart
sudo systemctl start tdr-server
sudo systemctl stop tdr-server
sudo systemctl restart tdr-server

# View logs
sudo journalctl -u tdr-server -f
```

## Troubleshooting

### Server Won't Start
1. Check if port 4000 is already in use:
   ```bash
   lsof -i :4000
   ```

2. Check logs for errors:
   ```bash
   tail -50 logs/server.log
   ```

3. Ensure virtual environment is activated:
   ```bash
   source env/bin/activate
   ```

### Server Crashes
1. Check system resources:
   ```bash
   free -h
   df -h
   ```

2. Look for Python errors:
   ```bash
   grep -i traceback logs/server.log
   ```

### Can't Connect
1. Check firewall:
   ```bash
   sudo ufw status
   ```

2. Test local connection:
   ```bash
   curl http://localhost:4000/api/ping
   ```

## Best Practices

1. **Always use the control script** rather than manual process management
2. **Monitor logs** regularly for warnings or errors
3. **Use quick_restart.sh** when deploying code changes
4. **Check server status** after any restart
5. **Keep logs rotated** to prevent disk space issues

## Quick Restart Workflow

The recommended workflow for code updates:

```bash
# 1. Stop client (on your local machine)
Ctrl+C in client terminal

# 2. Quick restart server (on server)
bin/quick_restart.sh

# 3. Restart client (on your local machine)
python src/tdr.py
```

This ensures:
- Latest code is deployed
- Data cache is used (fast restart)
- Clean connection state