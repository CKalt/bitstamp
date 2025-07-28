# Backtesting Plan for Tonight

## Live vs Development System Configuration Differences

### Live System (stable branch) expects best_strategy.json with:
```json
{
  "Frequency": "1H",
  "Strategy": "MA",
  "Short_Window": 6,      // Capital letters!
  "Long_Window": 34,      // Capital letters!
  "do_live_trades": true,
  "strategy_type": "MA",
  "enable_adaptive_strategy": false,
  "auto_resume": false,
  // Plus backtest result fields like:
  "Final_Balance": 11557.32,
  "Total_Return": 15.57,
  "Total_Trades": 41.0,
  "Sharpe_Ratio": 0.977
}
```

### Key Requirements for Live Compatibility:
1. **Capital Letters**: `Short_Window`, `Long_Window`, not lowercase
2. **Strategy Field**: Must be "MA" for MACrossoverStrategy
3. **do_live_trades**: Must be true for trading
4. **auto_resume**: Server forces true anyway

### Development System Differences:
- Has System Verifier integrated
- Respects auto_resume config setting
- Same MA parameter format requirements

## Tonight's Backtesting Approach

### 1. Direct Backtest with bktst.py
The current `src/bktst.py` reads best_strategy.json as config but outputs different format. We need to create a compatible output.

```bash
# Basic backtest command
python src/bktst.py \
  --start-window-days-back 120 \
  --end-window-days-back 0 \
  --high-frequency 1H \
  --low-frequency 15T \
  --initial 10000 \
  --ma-short 6 \
  --ma-long 34
```

### 2. Use run_comprehensive_backtest.py
This tests multiple MA combinations and finds the best:

```bash
python run_comprehensive_backtest.py
```

Output includes comparison table but needs format conversion.

### 3. Format Conversion Script Needed

Create a script to convert backtest output to live-compatible format:

```python
def convert_backtest_to_live_format(backtest_results, ma_short, ma_long):
    """Convert backtest results to live best_strategy.json format"""
    return {
        # Required MA parameters (CAPITAL LETTERS!)
        "Short_Window": ma_short,
        "Long_Window": ma_long,
        
        # Strategy configuration
        "Frequency": "1H",
        "Strategy": "MA",
        "Bar_Size": "1H",
        "do_live_trades": True,
        "strategy_type": "MA", 
        "enable_adaptive_strategy": False,
        "auto_resume": False,
        
        # Backtest results
        "Final_Balance": backtest_results.get('final_equity', 10000),
        "Total_Return": backtest_results.get('total_return', 0) * 100,
        "Total_Trades": backtest_results.get('num_trades', 0),
        "Average_Trades_Per_Day": backtest_results.get('trades_per_day', 0),
        "Profit_Factor": backtest_results.get('profit_factor', 1.0),
        "Sharpe_Ratio": backtest_results.get('sharpe_ratio', 0),
        
        # Optional fields from original
        "Last_Signal_Timestamp": 1749747600,
        "Last_Signal_Action": "GO SHORT",
        "Last_Trade_Timestamp": 1752078687,
        "Last_Trade_Price": 109337.0
    }
```

## Recommended Workflow for Tonight

1. **Test Current Settings First** (MA 6/34):
   ```bash
   python src/bktst.py --config best_strategy.json --data btcusd.log \
     --start-date 2025-01-01 --save-results current_ma_6_34.json
   ```

2. **Run Comprehensive Test**:
   ```bash
   python run_comprehensive_backtest.py
   ```
   This tests: 5/15, 8/21, 9/26, 10/46, 12/26, 15/30, 20/50, 21/55, 30/90, 50/100, 50/200

3. **Compare Strategies Visually**:
   ```bash
   python compare_strategies.py
   ```
   Creates strategy_comparison.png and detailed JSON

4. **Convert Best Result**:
   - Find best performing MA combination
   - Create proper best_strategy.json format
   - Validate all required fields present

5. **Deploy Safely**:
   ```bash
   # ON MAC (where you run backtests):
   # Backup current
   cp best_strategy.json best_strategy.json.backup_$(date +%Y%m%d_%H%M%S)
   
   # Copy new (after format conversion)
   cp converted_best_strategy.json best_strategy.json
   
   # Commit and push FROM MAC
   git add best_strategy.json
   git commit -m "Update MA parameters from backtesting"
   git push
   
   # ON SERVER (ssh ck):
   ssh ck
   gg btc  # or cd /home/chris/projects/bitstamp
   git pull
   
   # Restart trading with new parameters
   # (Follow server restart procedure)
   ```

## Data Verification

Before starting:
```bash
# Check data file
ls -lh btcusd.log  # Should be ~4.4GB

# Verify recent data
tail -1 btcusd.log | jq '.'

# Check data range
head -1 btcusd.log | jq '.timestamp' | xargs -I {} date -r {}
tail -1 btcusd.log | jq '.timestamp' | xargs -I {} date -r {}
```

## Critical Notes

1. **DO NOT** use the deployment script as-is - it outputs lowercase field names
2. **ENSURE** capital letters for Short_Window and Long_Window
3. **TEST** on development branch first if possible
4. **BACKUP** before any changes to best_strategy.json
5. **VERIFY** the format matches exactly what live system expects

## Alternative: Manual Parameter Update

If backtesting shows different optimal MA values (e.g., MA 10/20), you can manually update:

```bash
# Edit best_strategy.json
vim best_strategy.json

# Change only:
"Short_Window": 10,
"Long_Window": 20,

# Keep all other fields unchanged
```

This is safer than full replacement if you just want to test different MA values.