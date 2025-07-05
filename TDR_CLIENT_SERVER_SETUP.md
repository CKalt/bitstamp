# TDR Client-Server Setup Guide

This guide explains how to set up and run TDR in client-server mode, allowing you to run the trading engine on a remote server while controlling it from your local machine.

## Overview

The client-server architecture splits TDR into two components:
- **Server**: Pure execution engine that runs trading logic, WebSocket connections, and maintains logs
- **Client**: Provides the command-line interface and sends local configuration to the server

Key features:
- Configuration files (best_strategy.json) stay on the client side
- All logging happens server-side with REST endpoints to inspect logs
- The `.bitstamp` API credentials file remains secure on the server
- Claude Code integration works locally through the client
- Server only reads btcusd.csv log file from its local filesystem

## Prerequisites

1. Python 3.8+ on both client and server machines
2. All dependencies installed: `pip install -r requirements.txt`
3. Network connectivity between client and server (port 4000 by default)
4. SSH access to your remote server

## Server Setup (Remote Machine)

### 1. SSH to your remote server and set up environment:

```bash
ssh -L 4000:localhost:4000 ck
screen -S tdr
cd /home/chris/projects/bitstamp
source env/bin/activate
```

### 2. Install/update dependencies:

```bash
pip install flask flask-cors
# Or if you have updated requirements.txt:
pip install -r requirements.txt
```

### 3. Start the TDR server:

```bash
python src/tdr.py --server
# Or with custom port:
python src/tdr.py --server --port 5000
```

The server will start and display:
```
Starting TDR in SERVER mode on 0.0.0.0:4000
Server will wait for client to send configuration
Use --client flag on another instance to connect
```

Note: The server will NOT load any configuration files. It waits for the client to send configuration.

### 4. Detach from screen session:

Press `Ctrl-A` then `D` to detach from the screen session. The server will continue running.

To reattach later: `screen -r tdr`

## Client Setup (Local Machine)

### 1. Open a new terminal on your local machine:

```bash
cd /Users/chris/projects/python/btc
source env/bin/activate  # or your local virtual environment
```

### 2. Connect to the remote server:

```bash
python src/tdr.py --client
# Or if using a different server/port:
python src/tdr.py --client --server-url http://localhost:5000
```

You should see:
```
Starting TDR in CLIENT mode with configuration-based initialization
Loading configuration from best_strategy.json
Sending configuration to server...
✅ Successfully initialized TDR server at http://localhost:4000
```

The client will:
1. Load your local `best_strategy.json` file
2. Send it to the server for initialization
3. Load any saved position from `resume-auto-trade.json`
4. Start the interactive shell

## Using the Client

Once connected, you can use all the familiar TDR commands:

### Basic Commands:
```bash
# Check system status
status

# Get current price
get_price btcusd

# Buy/sell
buy btcusd 0.1
sell btcusd 0.1

# Start auto trading
auto_trade 1.5btc long

# Stop auto trading
stop_auto_trade

# Resume with saved position
resume_auto_trade 1.5btc long 108000
```

### Enhanced Client Commands:
```bash
# Check server connection
server

# Reconnect and reload configuration
reconnect

# View server logs
logs                    # Last 100 lines of server log
logs 50 server ERROR    # Last 50 lines with ERROR
logs 100 trading        # Trading logs (crypto_shell.log)

# View diagnostic events
show_diagnostics              # Recent diagnostic events
show_diagnostics SIGNAL_EVAL 30   # Last 30 signal evaluations
show_diagnostics TRADE_EXECUTED   # Recent trades

# View trade history from server
trades        # Last 10 trades
trades 50     # Last 50 trades

# Enable Claude Code integration (local)
enable_commands    # Start monitoring local commands/pending/
disable_commands   # Stop monitoring
```

## For Claude Code Integration

When using Claude Code with the client-server setup:

### 1. From your local terminal with Claude:

```bash
cd /home/chris/projects/bitstamp
claude
```

### 2. Enable command interface in the TDR client:

```bash
tdr> enable_commands
✅ Local command interface enabled for Claude Code
   Monitoring: commands/pending
   Processed: commands/processed
   Failed: commands/failed
```

### 3. Claude can now create command files locally:

The client monitors the local `commands/pending/` directory and forwards commands to the remote server.

### Example Claude workflow:
```python
# Claude creates command files in the LOCAL directory where the client is running
import json
import os
from datetime import datetime

# Create command for remote execution
command = {
    "timestamp": datetime.now().isoformat(),
    "command": "status",
    "source": "claude_check",
    "args": ""
}

# Save to LOCAL pending directory (where client is running)
os.makedirs("commands/pending", exist_ok=True)
with open("commands/pending/claude_status_check.json", "w") as f:
    json.dump(command, f)

# The client will:
# 1. Detect this file
# 2. Send the command to the remote server
# 3. Save the result to commands/processed/
```

### Key Points:
- Command files are created **locally** where the client runs
- The client forwards them to the remote server via REST
- Results are saved back to the local `commands/processed/` directory
- Claude can read the results from the local filesystem

## REST API Endpoints

The server exposes the following REST endpoints:

### Core Endpoints:
- `POST /api/initialize` - Initialize server with configuration (client does this automatically)
- `GET /api/status` - System status and position info
- `GET /api/ping` - Health check

### Information Endpoints:
- `GET /api/price/<symbol>` - Current price
- `GET /api/data/<symbol>?limit=100&frequency=1H` - Price data
- `GET /api/logs?lines=100&type=server&search=ERROR` - View logs
- `GET /api/diagnostics?event_type=ALL&count=50` - Diagnostic events
- `GET /api/trades?limit=50` - Trade history

### Command Execution:
- `POST /api/command` - Execute any shell command (primary interface)

### Example API calls:
```bash
# Initialize server (normally done by client)
curl -X POST http://localhost:4000/api/initialize \
  -H "Content-Type: application/json" \
  -d @best_strategy.json

# Get status
curl http://localhost:4000/api/status

# Execute command
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}'

# View logs
curl "http://localhost:4000/api/logs?lines=50&type=server&search=ERROR"

# Get diagnostics
curl "http://localhost:4000/api/diagnostics?event_type=SIGNAL_EVAL&count=20"
```

## Advanced Configuration

### Running Server Without SSH Tunnel:

If your server has a public IP or domain:

```bash
# On server:
python src/tdr.py --server --host 0.0.0.0 --port 4000

# On client:
python src/tdr.py --client --server-url http://your-server-ip:4000
```

### Security Considerations:

1. **Use SSH tunnels** for secure communication (as shown in basic setup)
2. **Firewall rules**: Only expose port 4000 to trusted IPs
3. **Authentication**: Consider adding API key authentication for production
4. **HTTPS**: Use a reverse proxy (nginx) with SSL for public deployments

### Environment Variables:

You can set the server URL via environment variable:
```bash
export TDR_SERVER_URL=http://localhost:4000
python src/tdr.py --client
```

## Troubleshooting

### Connection Issues:
```bash
# Check if server is running
curl http://localhost:4000/api/ping

# Check SSH tunnel
netstat -an | grep 4000

# Check server logs
# On server: check tdr_server.log
```

### Common Problems:

1. **"Cannot connect to server"**
   - Verify SSH tunnel is active
   - Check server is running
   - Verify port is not blocked

2. **Commands not working**
   - Use `reconnect` command
   - Check server logs for errors
   - Verify `best_strategy.json` exists on server

3. **WebSocket disconnected**
   - This is shown in status but doesn't affect REST API
   - Server will auto-reconnect WebSocket

## Performance Benefits

Running in client-server mode provides:
1. **Better performance**: Server can use more CPU/RAM
2. **Reliability**: Server continues running if client disconnects
3. **Multiple clients**: Multiple users can monitor/control
4. **Claude integration**: Can run Claude locally with more memory

## Monitoring

To monitor the server:
```bash
# View server logs
tail -f tdr_server.log

# Check system resources
htop

# Monitor in client
monitor 5  # Updates every 5 seconds
```

## Stopping the Server

To properly shut down:
```bash
# Reattach to screen
screen -r tdr

# Stop with Ctrl+C
# Exit screen
exit
```

Remember: Always use `stop_auto_trade` before shutting down if auto-trading is active!