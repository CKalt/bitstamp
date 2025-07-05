# TDR Client-Server Deployment Plan with Enhanced Monitoring

## Pre-Deployment Setup

### 1. Ensure Logging Directories Exist
Both client and server will create `logs/` directories automatically, containing:
- Server: `logs/tdr_server.log`
- Client: `logs/tdr_client.log`

## Enhanced Logging Features

### Log Format
```
2025-01-05 12:34:56,789 - [Component] - LEVEL - Message
```

### Log Tracking
- **[CMD-SEND]** - Client sending command to server
- **[CMD-EXEC]** - Server executing command
- **[CLAUDE-CMD]** - Commands from Claude via JSON files
- **Source tracking** - Shows "interactive", "claude", or custom source
- **Client IP** - Server logs show which client sent commands

## Deployment Steps with Monitoring

### Phase 1: Prepare Remote Server

#### Terminal 1 - Stop Current Trading
```bash
ssh chriskoin
screen -r tdr
(crypto) stop_auto_trade
(crypto) quit

# Commit changes
cd /home/chris/projects/bitstamp
git add .
git commit -m "Add client-server architecture with enhanced logging"
git push origin stable-added-adaptive-trad-n-chart-more
```

#### Terminal 2 - Monitor Server Logs
```bash
ssh chriskoin
cd /home/chris/projects/bitstamp
mkdir -p logs
tail -f logs/tdr_server.log
```

### Phase 2: Local Preparation

#### Terminal 3 - Update Local Code
```bash
cd /Users/chris/projects/python/btc
git pull origin stable-added-adaptive-trad-n-chart-more

# Create position resume file
cat > resume-auto-trade.json << 'EOF'
{
  "btc_amount": 1.52275326,
  "usd_amount": 0,
  "position": "long",
  "entry_price": 108234.0,
  "timestamp": "2025-01-05T00:58:21"
}
EOF

# Create logs directory
mkdir -p logs
```

#### Terminal 4 - Monitor Client Logs
```bash
cd /Users/chris/projects/python/btc
tail -f logs/tdr_client.log
```

### Phase 3: Start Server (Remote)

#### Terminal 1 - Start Server
```bash
# Back on remote server
screen -S tdr
cd /home/chris/projects/bitstamp
source env/bin/activate
python src/tdr.py --server --verbose  # verbose for detailed logging
```

You should see in Terminal 2 (server log):
```
2025-01-05 14:00:00,123 - [TDRServer] - INFO - Starting TDR Server on 0.0.0.0:4000
2025-01-05 14:00:00,124 - [TDRServer] - INFO - Waiting for client to send configuration...
```

### Phase 4: Start Client (Local)

#### Terminal 5 - SSH Tunnel
```bash
ssh -L 4000:localhost:4000 chriskoin
# Keep this running
```

#### Terminal 6 - Start Client
```bash
cd /Users/chris/projects/python/btc
source env/bin/activate
python src/tdr.py --client --verbose
```

You should see in Terminal 4 (client log):
```
2025-01-05 14:01:00,456 - [TDRClient] - INFO - Starting TDR Client - Server: http://localhost:4000, Config: best_strategy.json
2025-01-05 14:01:00,789 - [TDRClient] - INFO - [CMD-SEND] Source: initialization | Command: <config>
```

And in Terminal 2 (server log):
```
2025-01-05 14:01:00,790 - [TDRServer] - INFO - Initializing server with client configuration...
2025-01-05 14:01:01,123 - [TDRServer] - INFO - Server initialization complete
```

### Phase 5: Resume Trading & Enable Claude

In Terminal 6 (client shell):
```bash
tdr> status
tdr> resume_auto_trade 1.52275326btc long 108234
tdr> enable_commands
```

Monitor the logs to see:
- Client log: `[CMD-SEND] Source: interactive | Command: status`
- Server log: `[CMD-EXEC] Source: interactive | Client: 127.0.0.1 | Command: status`

### Phase 6: Test Claude Integration

#### Terminal 7 - Claude Test
```bash
cd /Users/chris/projects/python/btc
python claude_command_example.py
```

Watch the logs for:
- Client log: `[CLAUDE-CMD] Processing from claude_status_*.json | Source: claude | Command: status`
- Server log: `[CMD-EXEC] Source: claude | Client: 127.0.0.1 | Command: status`

## Log Monitoring Dashboard

For real-time monitoring, open these terminals side-by-side:

### Remote Server Side
```bash
# Terminal A - Server Logs
tail -f logs/tdr_server.log | grep -E "\[CMD-EXEC\]|\[ERROR\]|initialized|WebSocket"

# Terminal B - Trading Activity
tail -f crypto_shell.log | grep -E "TRADE|SIGNAL|POSITION|📊"

# Terminal C - Diagnostics
tail -f diagnostics_*.json | jq '.'
```

### Local Client Side
```bash
# Terminal D - Client Logs
tail -f logs/tdr_client.log | grep -E "\[CMD-SEND\]|\[CLAUDE-CMD\]|\[ERROR\]"

# Terminal E - Command Processing
watch -n 1 'ls -la commands/pending/ commands/processed/ commands/failed/ | tail -20'
```

## What to Look For

### Healthy Operation
- Commands show source (interactive/claude)
- Each client command has matching server execution
- Position updates after trades
- WebSocket remains connected
- No ERROR level messages

### Warning Signs
- Commands sent but not executed
- WebSocket disconnection messages
- ERROR level logs
- Commands stuck in pending/
- Mismatched position between client/server

## Example Log Flows

### Interactive Command:
```
CLIENT: [CMD-SEND] Source: interactive | Command: status
SERVER: [CMD-EXEC] Source: interactive | Client: 127.0.0.1 | Command: status
```

### Claude Command:
```
CLIENT: [CLAUDE-CMD] Processing from claude_buy_12345.json | Source: claude | Command: buy btcusd 0.1
CLIENT: [CMD-SEND] Source: claude | Command: buy btcusd 0.1
SERVER: [CMD-EXEC] Source: claude | Client: 127.0.0.1 | Command: buy btcusd 0.1
```

### Auto-Trade Signal:
```
SERVER: 📊 Adaptive strategy evaluation #5 at 2025-01-05 14:15:00
SERVER: 📊 TRENDING: MA crossover signal detected
SERVER: Executing LONG trade for 1.52275326 BTC
```

## Troubleshooting with Logs

### Client Can't Connect
Check client log for:
```
ERROR - Cannot connect to server
```
Solution: Verify SSH tunnel is active

### Commands Not Processing
Check if commands move from pending to processed/failed
Check server log for execution

### Position Mismatch
Compare position in client vs server status
Check for failed trades in server log

This enhanced logging system provides complete visibility into the command flow, making it easy to diagnose issues and monitor the system's health.