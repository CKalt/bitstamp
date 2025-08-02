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

## Development Workflow

### The gg Navigation System
The `gg` system provides consistent navigation shortcuts across both Mac and EC2:

- `gg btc` → Navigate to live/production directory
  - Mac: `/Users/chris/projects/python/btc`
  - EC2: `/home/chris/projects/bitstamp`
- `gg tst` → Navigate to test/development directory
  - Mac: `/Users/chris/projects/python/btc-testing`
  - EC2: `/home/chris/projects/bitstamp-testing`

**Note**: On EC2, you must first run `source ~/ggmap` to enable gg commands.

### Git Branch Structure
Both Mac and EC2 maintain the same branch structure:

- **At `gg btc` location**: `stable-added-adaptive-trad-n-chart-more` branch (production)
- **At `gg tst` location**: `development` branch (testing)

**Important**: The production branch name is historical and should eventually be merged into `main`.

### Symbolic Links for Claude
To facilitate Claude's workflow, symbolic links exist in the main directory:

**Mac**:
```bash
/Users/chris/projects/python/btc/tst -> /Users/chris/projects/python/btc-testing
```

**EC2**:
```bash
/home/chris/projects/bitstamp/tst -> /home/chris/projects/bitstamp-testing
```

This allows Claude to work on test code by simply doing `cd tst` from the main directory.

### Development Rules

1. **All code changes made on Mac**
   - Claude makes changes in the Mac environment
   - Can freely navigate using `cd tst` for test development
   - Commits and pushes to GitHub

2. **EC2 is for running only**
   - Claude can SSH to check status: `ssh ck`
   - Must ask permission before modifying files on EC2
   - Deployment is via git pull only

3. **Deployment Flow**
   ```
   Mac Development → Git Push → [Human Action] → EC2 Git Pull → Restart Services
   ```

### Screen Session Workflow

**Required screens for full system operation**:

1. **EC2 Server Side**:
   - `screen -S btc` - WebSocket price feed (must run first, shared by all)
   - `screen -S server` - Live trading server (port 4000)
   - `screen -S server-tst` - Test trading server (port 4002)

2. **Mac Client Side**:
   - `screen -S claude-tdr` - Claude development session
   - `screen -S client-tdr` - Live trading client (connects to localhost:4000)
   - `screen -S client-tst` - Test trading client (connects to localhost:4002)

3. **SSH Tunnels** (automatic via autossh):
   - Must be running for Mac clients to reach EC2 servers
   - Check with `fix-tunnels` if connection issues

### Command Location Context
When giving commands, always specify:
- **"ON MAC"** - for local development commands
- **"ON SERVER"** or **"ON EC2"** - for remote server commands

### Typical Development Session

1. **Start on Mac**: `gg btc` (you're in production directory)
2. **Work on test code**: `cd tst` (now in development branch)
3. **Make changes, test locally**
4. **Commit and push**: 
   ```bash
   git add -A
   git commit -m "Description of changes"
   git push origin development
   ```
5. **Request deployment**: "Please deploy these changes to EC2 test server"
6. **Human deploys**:
   ```bash
   ssh ck
   source ~/ggmap && gg tst
   git pull origin development
   # Restart test server
   ```