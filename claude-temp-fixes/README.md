# Position Tracking Fix Scripts

These scripts help fix position tracking mismatches in the TDR trading system.

## Scripts Overview

### 1. `analyze_position.py`
Analyzes the current position state by examining:
- trades.json (actual trades executed)
- resume-auto-trade.json (saved position state)
- best_strategy.json (configuration)

Shows any discrepancies between actual and tracked positions.

**Usage:**
```bash
source source-venv.sh
python claude-temp-fixes/analyze_position.py
```

### 2. `fix_position.py`
Automatically fixes position tracking to match actual trades:
- Calculates real position from trades.json
- Updates resume-auto-trade.json with correct values
- Updates best_strategy.json with correct entry price
- Creates backups of modified files

**Usage:**
```bash
source source-venv.sh
python claude-temp-fixes/fix_position.py
```

### 3. `prepare_short_resume.py`
Manually sets up a SHORT position with specified parameters:
- Creates resume-auto-trade.json for SHORT position
- Updates best_strategy.json
- Does NOT execute any trades

**Usage:**
```bash
source source-venv.sh
python claude-temp-fixes/prepare_short_resume.py 172000 117987
```

## Current Situation

Based on trades.json, you are currently LONG with ~1.436 BTC (bought at 13:00 UTC).
The system thinks you're SHORT with $172,000 USD.

## Your Options

### Option 1: Fix to Match Reality (Recommended)
Run `fix_position.py` to update tracking to match your actual LONG position.
```bash
python claude-temp-fixes/fix_position.py
```

### Option 2: Force SHORT Position
If you want to be SHORT at $172,000 with entry at $117,987:
1. First, manually sell your BTC
2. Run: `python claude-temp-fixes/prepare_short_resume.py 172000 117987`
3. Restart the server

### Option 3: Let System Trade Naturally
Just restart the server and let it trade when MAs cross. The position tracking
will self-correct after the next trade.

## Important Notes

- These scripts run on the server in `/home/chris/projects/bitstamp`
- Always run `source source-venv.sh` first to activate the virtual environment
- The scripts create timestamped backups before modifying any files
- After running a fix script, restart the server for changes to take effect