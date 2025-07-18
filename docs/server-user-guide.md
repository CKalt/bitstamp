# TDR Server User Guide

## Overview

The TDR Server is the backend component of the Bitcoin trading system that manages data, executes trades, and processes commands. It runs on port 4000 and communicates with clients via HTTP API and WebSocket connections.

## Server Architecture

### Single Service Design
The server runs as a single Flask application (`src/tdr_server.py`) that handles:
- HTTP API endpoints for commands and status
- WebSocket connections for real-time updates
- Background data loading and processing
- Trade execution and strategy management

## Data Loading Optimization

### The Problem
Previously, the server would load and process the entire `btcusd.log` file (1.1+ million records) on every startup, which could take several minutes and block all operations.

### The Solution: Background Loading with Caching

#### 1. **Asynchronous Background Loading**
- Server starts immediately and accepts connections
- Historical data loads in a background thread
- Trading commands are queued until data is ready
- Status commands work immediately

#### 2. **Pickle Cache System**
The server implements a smart caching system to avoid reprocessing data:

```python
# Cache location
data_cache/btcusd_processed.pkl
```

**How it works:**
- First run: Processes `btcusd.log` and saves to `.pkl` file
- Subsequent runs: Checks if cache is newer than source file
- If cache is valid: Loads in ~2 seconds instead of minutes
- If cache is stale: Reprocesses and updates cache

#### 3. **Cache Validation**
The cache is considered valid when:
- Cache file exists
- Cache file modification time > source file modification time
- No errors during cache load

### Startup Process

1. **Server Initialization** (Immediate)
   ```
   Starting TDR Server on 0.0.0.0:4000
   Waiting for client to send configuration...
   ```

2. **Client Connection**
   - Client sends `best_strategy.json` configuration
   - Server initializes with strategy parameters
   - Background data loading begins

3. **Background Data Loading**
   ```
   USE_DATA_CACHE=1  # Uses cache if valid
   USE_DATA_CACHE=0  # Forces fresh load
   ```

4. **Progressive Availability**
   - Status endpoints: Available immediately
   - Historical queries: Available after data loads
   - Trading: Blocked until data ready

## Quick Restart Script

The `bin/quick_restart.sh` script handles:
1. Graceful server shutdown
2. Git pull for updates
3. Cache validation
4. Server restart with appropriate caching

```bash
# Example output
✅ Cache is valid
   Using cached data for fast startup
```

## Server Commands

### Via Client
All commands are sent through the client, which forwards them to the server:
```
status          # Current position and performance
signals         # Trading signals and indicators
history_status  # Data loading progress
auto_trade      # Start automated trading
```

### Direct API Access (for debugging)
```bash
# Check if server is running
curl http://localhost:4000/api/ping

# Get server status
curl http://localhost:4000/api/status

# Send command
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "status"}'
```

## Performance Metrics

### Before Optimization
- Startup time: 3-5 minutes
- Memory usage: High during processing
- Client blocked during entire startup

### After Optimization
- Startup time: <5 seconds (with cache)
- Initial response: Immediate
- Full data availability: ~30 seconds (from cache)
- Fresh data processing: Still 3-5 minutes (first run only)

## Data Flow

1. **Source Data**: `btcusd.log` (chronological trade records)
2. **Processing**: Parses timestamps, prices, validates data
3. **Cache Storage**: `data_cache/btcusd_processed.pkl`
4. **Memory Structure**: Pandas DataFrame with datetime index
5. **Access Pattern**: Time-series queries for technical analysis

## Monitoring Server Health

### Check Data Loading Status
```
history_status
```

Shows:
- Loading phase (parsing/processing/complete)
- Record count
- Progress percentage

### Check Cache Status
```bash
ls -la data_cache/
# Shows cache file size and modification time
```

### Server Logs
```bash
tail -f logs/server.log
tail -f logs/server_restart.log
```

## Troubleshooting

### Server Won't Start
1. Check if port 4000 is already in use:
   ```bash
   lsof -i :4000
   ```

2. Kill stuck processes:
   ```bash
   pkill -f "python.*tdr_server"
   ```

### Slow Startup
- Check if cache exists: `ls data_cache/`
- Force cache rebuild: `rm data_cache/*.pkl`
- Check source file size: `ls -lh btcusd.log`

### Data Not Loading
- Check `history_status` command
- Review server logs for errors
- Ensure `btcusd.log` is not corrupted

## Configuration

The server receives its configuration from the client's `best_strategy.json`, including:
- Trading strategy parameters
- Pivot protection settings
- Risk management limits
- Technical indicator windows

## Security Notes

- Server binds to `0.0.0.0:4000` (all interfaces)
- No authentication (designed for local/tunnel use)
- Commands are executed with server privileges
- Keep behind firewall or use SSH tunnels for remote access