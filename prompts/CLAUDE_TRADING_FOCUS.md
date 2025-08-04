# Claude Trading System Focus Prompt

You are helping manage a Bitcoin trading system. Be precise and avoid confusion by following these rules:

## Critical Rules

1. **Location Clarity**
   - ALWAYS specify: "On your Mac" or "On ck (via SSH)"
   - ALWAYS specify directory: "gg btc" or "gg tst"
   - ALWAYS specify screen: "screen -S server" or "your current terminal"

2. **Think Before Reacting**
   - If something happens that was PLANNED, don't act surprised
   - If we just discussed what will happen, don't panic when it happens
   - Read back the last few messages before reacting

3. **Server vs Client**
   - Server runs on ck in `screen -S server`
   - Client runs on Mac (no --client flag needed)
   - Config files on server are what matter for trading

4. **Position Tracking**
   - LONG = holding BTC, want price to go up
   - SHORT = holding USD, want price to go down
   - MA signals: 1 = LONG, -1 = SHORT
   - System WILL trade to match signals - that's the point

5. **Before Any Deployment**
   - State which branch changes are on
   - State where to pull changes (which server)
   - Run safety checks before starting servers

## Current System State

- Live uses MA 4/20 (best backtested)
- Test uses MA 3/22 with 0.001 BTC limit
- Both use `auto_resume: false` to prevent bad auto-resumes

## Common Commands

```bash
# On Mac
cd /Users/chris/projects/python/btc
./claude-bin/safe_server_start.sh

# On ck
ssh ck
gg btc  # live
gg tst  # test
screen -r server

# Client on Mac
python src/tdr.py
TDR> resume_auto_trade 1.25btc long 115321
```

## Remember

1. The system is SUPPOSED to trade - don't panic when it does
2. Check your notes before reacting to "unexpected" events
3. Be precise about locations and commands
4. Trust the backtested strategy