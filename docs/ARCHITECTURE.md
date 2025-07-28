# System Architecture

## Overview

The Bitcoin trading system consists of backtesting components running locally on Mac and live trading servers running on EC2. All code deployment flows through GitHub.

## Component Locations

### Local Mac Environment
- **Purpose**: Development, backtesting, and strategy optimization
- **Directory**: `/Users/chris/projects/python/btc` (main)
- **Directory**: `/Users/chris/projects/python/btc-testing` (development branch)
- **Data**: btcusd.log (4.4GB of historical tick data)
- **Access**: Direct file system access

### Remote EC2 Environment
- **Purpose**: Live and test trading servers
- **Access**: `ssh ck`
- **Live Directory**: `/home/chris/projects/bitstamp`
- **Test Directory**: `/home/chris/projects/bitstamp-testing`
- **Instance Type**: t3.large (8GB RAM)

## Data Flow Architecture

```
┌─── LOCAL MAC ───┐         ┌─── GITHUB ───┐         ┌─── EC2 SERVER ───┐
│                 │         │              │         │                  │
│  Development    │  push   │  Repository  │  pull   │  Trading Server  │
│  & Backtesting  ├────────►│              ├────────►│                  │
│                 │         │              │         │                  │
└─────────────────┘         └──────────────┘         └──────────────────┘
```

## Network Architecture

### SSH Tunnels (autossh)
Tunnels make remote EC2 ports appear as localhost:

- `localhost:4000` → `ck:4000` (Live trading server)
- `localhost:8050` → `ck:8050` (Live charts)
- `localhost:4002` → `ck:4002` (Test trading server)
- `localhost:8051` → `ck:8051` (Test charts)

Managed via `~/projects/tunnels/tunnels.json`

### Trading System Ports
- **4000**: Live TDR server API
- **4002**: Test TDR server API
- **8050**: Live Dash charts
- **8051**: Test Dash charts

## Directory Structure

### Mac (Local)
```
/Users/chris/projects/python/
├── btc/                    # Main development (stable branch)
│   ├── src/               # Source code
│   ├── docs/              # Documentation
│   ├── scripts/           # Utility scripts
│   ├── btcusd.log         # Historical data (4.4GB)
│   └── best_strategy.json # Trading configuration
│
└── btc-testing/           # Development branch
    ├── src/               # Same structure
    ├── btcusd.log → ../btc/btcusd.log  # Symlink
    └── env/               # Virtual environment
```

### EC2 (Remote)
```
/home/chris/projects/
├── bitstamp/              # Live trading
│   ├── src/              
│   ├── logs/             
│   ├── btcusd.log        # Live price feed
│   └── best_strategy.json
│
└── bitstamp-testing/      # Test trading
    ├── src/              
    ├── logs/             
    ├── btcusd.log → ../bitstamp/btcusd.log
    └── best_strategy.json # With position limits
```

## Process Architecture

### Screen Sessions

**Mac (Local)**:
- `claude-tdr`: Claude Code development session
- `client-tdr`: Live trading client
- `client-tst`: Test trading client

**EC2 (Remote)**:
- `btc`: WebSocket price feed (shared)
- `server`: Live trading server
- `server-tst`: Test trading server

### Trading System Components

1. **Price Feed** (`websock-ticker2.py`)
   - Runs once, shared by all
   - Writes to btcusd.log
   - Real-time Bitstamp data

2. **Trading Server** (`tdr.py --server`)
   - Loads historical data on startup
   - Processes trading signals
   - Executes trades via Bitstamp API

3. **Trading Client** (`tdr.py`)
   - Connects to server
   - Displays positions and charts
   - Sends commands to server

## Git Workflow

### Branches
- `stable-added-adaptive-trad-n-chart-more`: Production branch
- `development`: Testing branch

### Deployment Process
1. Develop on Mac in appropriate directory
2. Test locally with backtesting
3. Commit and push to GitHub
4. SSH to EC2 server
5. Pull changes from GitHub
6. Restart affected services

### NEVER Do These
- Copy files directly to server (use git)
- Run trading commands locally
- Access server files without SSH
- Modify production without testing

## Configuration Management

### best_strategy.json
- Controls trading parameters
- Must be deployed via git
- Test version includes position limits:
  ```json
  {
    "max_position_btc": 0.001,
    "max_position_usd": 100,
    "trading_mode": "development"
  }
  ```

### Environment Variables
- Set via `source source-venv.sh`
- Python virtual environments separate prod/dev

## Memory Constraints

EC2 t3.large instance (8GB RAM):
- Price feed: ~500MB
- Live server: ~1.5GB after loading history
- Test server: ~1.5GB (can be stopped if needed)
- System overhead: ~1GB

## Security Notes

- `.bitstamp` files contain API credentials
- Never read or expose these files
- All trading requires explicit `do_live_trades: true`
- Test environment limited to 0.001 BTC ($100)