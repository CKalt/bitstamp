# Screen Architecture for TDR Trading System

## Overview

We use screen sessions to manage all components of the trading system. This allows persistent sessions that survive SSH disconnections.

## Screen Naming Convention

### Local Machine (Mac)
- `claude-tdr` - Claude Code session (you are here!)
- `client-tdr` - Live TDR client connecting to server
- `client-tst` - Test TDR client (0.001 BTC) connecting to server-tst

### Server (ck)
- `ticker` - Price feed (websock-ticker2.py) - shared by both live and dev
- `server` - Live trading server (port 4000)
- `server-tst` - Test/dev trading server (port 4001)

## Directory Structure with ggmap

### Local Machine
```
gg btc  → /Users/chris/projects/python/btc         (live)
gg tst  → /Users/chris/projects/python/btc-testing (test)
```

### Server
```
gg btc  → /home/chris/projects/bitstamp            (live)
gg tst  → /home/chris/projects/bitstamp-testing    (test)
```

## Architecture Diagram

```
┌─────────────── LOCAL MACHINE ───────────────┐     ┌─────────────── SERVER (ck) ─────────────────┐
│                                              │     │                                             │
│  ┌─────────────┐        ┌─────────────┐     │     │     ┌─────────────┐                         │
│  │ claude-tdr  │        │ client-tdr  │     │     │     │   ticker    │                         │
│  │   (you!)    │        │   (live)    │────────────────▶│(price feed) │                         │
│  └─────────────┘        │  gg btc     │     │     │     └──────┬──────┘                         │
│                         └─────────────┘     │     │            │                                │
│                                              │     │            ▼                                │
│                         ┌─────────────┐     │     │     ┌─────────────┐     ┌─────────────┐    │
│                         │ client-tst  │     │     │     │   server    │     │ server-tst  │    │
│                         │   (test)    │────────────────▶│  (live)     │     │   (dev)     │    │
│                         │  gg tst     │     │     │     │ port 4000   │     │ port 4001   │    │
│                         └─────────────┘     │     │     │  gg btc     │     │  gg tst     │    │
│                                              │     │     └─────────────┘     └─────────────┘    │
└──────────────────────────────────────────────┘     └─────────────────────────────────────────────┘
```

## Price Feed Sharing

Both `server` and `server-tst` share the same price feed via symlink:
```
/home/chris/projects/bitstamp/btcusd.log         (actual file, written by ticker)
/home/chris/projects/bitstamp-testing/btcusd.log → ../bitstamp/btcusd.log (symlink)
```

## Common Commands

### Starting Services (Server)

```bash
# Start price feed (only one needed)
ssh ck
gg btc
screen -S ticker python src/websock-ticker2.py

# Start live server
gg btc
screen -S server python src/tdr.py --server

# Start test server
gg tst
screen -S server-tst python src/tdr.py --server --port 4001
```

### Starting Clients (Local)

```bash
# Start live client
gg btc
screen -S client-tdr python src/tdr.py

# Start test client
gg tst
screen -S client-tst python src/tdr.py
```

### Monitoring

```bash
# On server
ssh ck "screen -list"

# On local
screen -list

# Check all systems
ssh ck "cd /home/chris/projects/bitstamp-testing && ./monitor-screens.sh"
```

### Attaching to Sessions

```bash
# Local
screen -r claude-tdr    # This Claude session
screen -r client-tdr    # Live client
screen -r client-tst    # Test client

# Server
ssh ck
screen -r ticker        # Price feed
screen -r server        # Live server
screen -r server-tst    # Test server
```

### Detaching from Sessions
Always use `Ctrl+A, D` to detach from a screen session without killing it.

## Trading Commands

### Live Trading (Full Position)
```bash
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 1.444btc MA short=6 long=34 do_live_trades=True hist_position=long"}'
```

### Test Trading (0.001 BTC)
```bash
curl -X POST http://localhost:4001/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "auto_trade 0.001btc MA short=6 long=34 do_live_trades=True hist_position=long"}'
```

## Best Practices

1. **Always use screen** - Never run trading processes directly in SSH
2. **One price feed** - Only run ticker once in live directory
3. **Check before starting** - Use `screen -list` to avoid duplicates
4. **Use ggmap** - Always use `gg btc` or `gg tst` for navigation
5. **Separate logs** - Live uses `tdr_server.log`, test uses `tdr_server_dev.log`

## Emergency Procedures

### Stop All Trading
```bash
# Stop live
curl -X POST http://localhost:4000/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'

# Stop test
curl -X POST http://localhost:4001/api/command \
  -H "Content-Type: application/json" \
  -d '{"command": "stop_auto_trade"}'
```

### Kill Screens
```bash
# Kill specific screen
screen -S [name] -X quit

# Kill all screens (careful!)
killall screen
```

## Development Workflow

1. Make changes in test directory (`gg tst`)
2. Test with 0.001 BTC position
3. Monitor for errors and correct behavior
4. Once verified, cherry-pick or merge to live
5. Deploy to live with confidence

This architecture provides safe testing with real money while limiting risk to ~$100.