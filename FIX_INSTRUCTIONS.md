# Fix Instructions for SHORT Position Entry Price

## Issue Summary
- System shows SHORT entry at $118,198 (old LONG entry) instead of $116,970 (actual SELL price)
- Manual SELL trade wasn't recorded in trades.json
- P&L calculations are incorrect

## Fix Steps (Run on Server)

### 1. Copy Scripts to Server
```bash
scp add_sell_trade.py fix_short_position.py chris@your-server:/home/chris/projects/bitstamp/
```

### 2. SSH to Server and Run Scripts
```bash
ssh chris@your-server
cd /home/chris/projects/bitstamp

# First, add the missing SELL trade
python3 add_sell_trade.py

# Then fix the position tracking
python3 fix_short_position.py
```

### 3. Restart Auto-Trader
```bash
# Stop current auto-trader
pkill -f auto_trade.py

# Resume with correct SHORT position
python3 auto_trade.py resume 170090.79usd short 116970
```

## What These Scripts Do

### add_sell_trade.py
- Adds the missing SELL trade at $116,970 to trades.json
- Records: 1.45378686 BTC sold for $170,090.79

### fix_short_position.py
- Updates resume-auto-trade.json with:
  - Correct SHORT entry price: $116,970
  - Proper position tracking values
  - Accurate P&L calculations

## Verification
After running, check status:
```bash
python3 auto_trade.py status long
```

Should show:
- Position: Short
- Entry Price: $116,970.00 (not $118,198)
- Correct P&L based on SHORT entry