# Auto-Resume Flow Documentation

## Overview
Auto-resume is a feature that saves the current trading position to disk and automatically restores it when the server restarts. This prevents losing track of positions during server restarts.

## How It Works

### 1. When Position State is Saved

The `save_resume_state()` function is called:
- **After each trade execution** (unless it's part of a multi-part trade)
- **When auto_trade command is started** 
- **Periodically during auto-trading** (in the monitor_positions loop)
- **When stop_auto_trade is called**

### 2. What Gets Saved

The system saves to `resume-auto-trade.json`:
```json
{
  "timestamp": "2025-07-26T21:45:00",
  "position": "LONG",                     // or "SHORT"
  "amount": 0.001,                        // Position size
  "unit": "btc",                          // "btc" for long, "usd" for short
  "entry_price": 118006.00,               // Entry price
  "current_price": 118500.00,             // Current market price
  "unrealized_pnl": 0.49,                 // Unrealized profit/loss
  "command": "resume_auto_trade 0.001btc long 118006",  // Resume command
  "strategy": {
    "type": "MACrossoverStrategy",
    "short_window": 6,
    "long_window": 34
  },
  "balances": {
    "btc": 0.001,
    "usd": 9881.994
  },
  "trades_executed": 1,
  "last_trade_time": "2025-07-26T21:45:00",
  "trade_references": [...],              // References to trades.json entries
  "pivot_protection": {...}               // Pivot protection state if enabled
}
```

### 3. When Auto-Resume Loads

The loading happens in `auto_resume_trading()` function which is called:
- **After historical data loads successfully** (once per server start)
- **Only if `auto_resume` is True in config**

The process:
1. Check if `best_strategy.json` has `"auto_resume": true`
2. If false, skip auto-resume entirely (our fix!)
3. If true, check if `resume-auto-trade.json` exists
4. If exists, load the file and extract the command
5. Execute: `shell.do_resume_auto_trade("0.001btc long 118006")`

### 4. What the Resume Command Does

`resume_auto_trade` command (in shell.py):
1. Validates the position format (amount, unit, position type)
2. Sets initial balances based on the saved position
3. Starts auto_trade with the saved parameters
4. The strategy resumes from the saved position and entry price

## Key Conditions

### Auto-Resume WILL Run When:
- `"auto_resume": true` in best_strategy.json
- `resume-auto-trade.json` file exists
- Server restarts and loads historical data

### Auto-Resume WON'T Run When:
- `"auto_resume": false` in best_strategy.json (our fix!)
- No `resume-auto-trade.json` file exists
- Historical data fails to load
- Server started with `--wait-for-client` flag

## File Locations
- Resume file: `/path/to/project/resume-auto-trade.json`
- Config: `/path/to/project/best_strategy.json`
- Position history: `/path/to/project/position-history.json`
- Trade log: `/path/to/project/trades.json`

## Why It Was Forcing True

Before our fix, two places were forcing auto_resume:
1. **tdr_server.py line 594**: When `resume_auto_trade` command was sent while history was loading
2. **tdr_server_standalone.py line 132**: During standalone initialization

This meant even if you set `"auto_resume": false` in config, it would be ignored!

## Our Fix

We changed:
1. **tdr_server.py**: Added check for config setting before attempting auto-resume
2. **tdr_server_standalone.py**: Removed the line forcing auto_resume = True

Now the system properly respects the `auto_resume` config setting.