# Theoretical Trades in the TDR System - Summary

## Overview
Theoretical trades are a mechanism in the TDR system to track positions when no actual trades have been executed. They represent what the position *should* be worth based on the user's initial position declaration.

## 1. Where Theoretical Trades are Created and Set

### Main Creation Points:

#### A. In `shell.py` - `do_auto_trade()` method (lines 570-734)
This is where theoretical trades are primarily created during system initialization:

1. **Matching Positions (Cases 1 & 3)** - When user's desired position matches system recommendation:
   - **Long Position** (lines 579-606): Creates theoretical long trade
   - **Short Position** (lines 609-638): Creates theoretical short trade

2. **Mismatched Positions with auto_align=False** (lines 694-734):
   - When user position differs from MA signal but auto_align_position is False
   - Creates theoretical trade to track user's position without immediate reversal

3. **Initial Position Setup** (lines 641-680):
   - Creates theoretical trades when initializing with matching positions
   - Sets up position tracking with theoretical entry prices

### Theoretical Trade Structure:
```python
theoretical_trade = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'direction': 'long' or 'short',
    'amount': amount_num,  # BTC amount for long, USD amount for short
    'entry_price': effective_entry_price,
    'theoretical': True
}
```

## 2. How Theoretical Trades are Used in Status Display

### In `strategies.py` - `get_status()` method (lines 1621-1636):
- When `trades_executed == 0` and `theoretical_trade` exists
- Calculates and displays position info based on theoretical trade:
  - Entry price from theoretical trade
  - Position size based on theoretical amount
  - Unrealized P&L calculated from theoretical entry vs current price

### In `shell.py` - `do_status()` method (lines 1413-1419):
- Displays theoretical trade information in status output
- Shows timestamp, direction, amount, and theoretical flag
- Only shown when no actual trades have been executed

## 3. When Theoretical Trades are Cleared

### Primary Clearing Points:

1. **After Actual Trade Execution** (`strategies.py`, lines 1055-1059):
   - Cleared in `execute_trade()` method after any successful trade
   - Ensures theoretical trades don't persist once real trading begins

2. **When Loading Real Trades** (`shell.py`, line 522):
   - Cleared when system loads existing trades from trades.json
   - Prevents theoretical trades from overriding real position data

3. **Server Position Updates** (`tdr_server.py`, lines 830-832):
   - Cleared when server updates position with actual trading data
   - Ensures consistency between server and client state

## 4. Relationship Between Theoretical and Actual Trades

### Key Relationships:

1. **Mutually Exclusive**:
   - System uses either theoretical OR actual trades, never both
   - Theoretical trades only exist when `trades_executed == 0`

2. **Position Tracking**:
   - Theoretical trades provide initial position tracking
   - Replaced by actual position data once trading begins
   - Entry price calculation differs:
     - Theoretical: Uses declared/market price at initialization
     - Actual: Uses cost-basis tracking from executed trades

3. **Status Display Logic**:
   - If no trades executed AND theoretical trade exists → Show theoretical position
   - If trades executed → Show actual position from real trades
   - Never shows both simultaneously

4. **Trade Execution Impact**:
   - First real trade immediately clears theoretical trade
   - Position tracking switches to cost-basis method
   - Entry price calculated from actual trade prices

## Key Code Locations:
- Creation: `src/tdr_core/shell.py` (lines 570-734)
- Status Display: `src/tdr_core/strategies.py` (lines 1621-1636)
- Status Command: `src/tdr_core/shell.py` (lines 1413-1419)
- Clearing: `src/tdr_core/strategies.py` (line 1059), `src/tdr_core/shell.py` (line 522)
- Server Sync: `src/tdr_server.py` (lines 830-832)