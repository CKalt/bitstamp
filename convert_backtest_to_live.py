#!/usr/bin/env python3
"""
Convert backtest results to live-compatible best_strategy.json format
Ensures proper capitalization and required fields for production system
"""
import json
import sys
from datetime import datetime
import argparse


def convert_backtest_to_live_format(backtest_file, ma_short, ma_long, output_file=None):
    """
    Convert backtest results to live best_strategy.json format
    
    CRITICAL: Live system expects capital letters (Short_Window, Long_Window)
    """
    # Load backtest results
    with open(backtest_file, 'r') as f:
        backtest_results = json.load(f)
    
    # Extract metrics - handle different possible field names
    metrics = backtest_results.get('metrics', {})
    summary = metrics.get('summary', {})
    risk = metrics.get('risk', {})
    win_loss = metrics.get('win_loss_analysis', {})
    
    # Build live-compatible format
    live_format = {
        # MA parameters - MUST BE CAPITAL LETTERS!
        "Short_Window": int(ma_short),
        "Long_Window": int(ma_long),
        
        # Required strategy configuration
        "Frequency": "1H",
        "Strategy": "MA",
        "Bar_Size": "1H",
        "do_live_trades": True,  # Required for live trading
        "strategy_type": "MA",
        "enable_adaptive_strategy": False,
        "auto_resume": False,  # Server forces true anyway
        
        # Backtest performance metrics
        "Final_Balance": float(backtest_results.get('final_equity', 
                                                    backtest_results.get('final_value', 10000))),
        "Total_Return": float(summary.get('total_return_pct', 
                                         backtest_results.get('total_return', 0) * 100)),
        "Total_Trades": float(backtest_results.get('num_trades', 0)),
        "Average_Trades_Per_Day": float(backtest_results.get('trades_per_day', 
                                                             backtest_results.get('num_trades', 0) / 120)),
        "Profit_Factor": float(win_loss.get('profit_factor', 1.0)),
        "Sharpe_Ratio": float(risk.get('sharpe_ratio', 0)),
        
        # Optional fields from original format
        "Last_Signal_Timestamp": 1749747600,  # Placeholder
        "Last_Signal_Action": "GO SHORT",     # Placeholder
        "Last_Trade_Timestamp": 1752078687,   # Placeholder
        "Last_Trade_Price": 109337.0          # Placeholder
    }
    
    # Save to file
    if not output_file:
        output_file = f"best_strategy_ma_{ma_short}_{ma_long}.json"
    
    with open(output_file, 'w') as f:
        json.dump(live_format, f, indent=2)
    
    print(f"✅ Created live-compatible config: {output_file}")
    print(f"   MA Strategy: {ma_short}/{ma_long}")
    print(f"   Sharpe Ratio: {live_format['Sharpe_Ratio']:.3f}")
    print(f"   Total Return: {live_format['Total_Return']:.2f}%")
    print(f"   Profit Factor: {live_format['Profit_Factor']:.2f}")
    
    return live_format


def validate_live_format(config):
    """Validate the configuration has all required fields for live system"""
    required_fields = [
        "Short_Window", "Long_Window", "Frequency", "Strategy", 
        "do_live_trades", "strategy_type"
    ]
    
    missing = [field for field in required_fields if field not in config]
    if missing:
        print(f"❌ WARNING: Missing required fields: {missing}")
        return False
    
    # Check capitalization
    if "short_window" in config or "long_window" in config:
        print("❌ WARNING: Found lowercase window parameters - live system needs capitals!")
        return False
    
    # Check do_live_trades
    if not config.get("do_live_trades"):
        print("❌ WARNING: do_live_trades is not True - trades won't execute!")
        return False
    
    print("✅ Configuration valid for live system")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert backtest results to live best_strategy.json format"
    )
    parser.add_argument(
        "backtest_file",
        help="Path to backtest results JSON file"
    )
    parser.add_argument(
        "--ma-short", 
        type=int, 
        help="Short MA window (e.g., 6) - required for conversion"
    )
    parser.add_argument(
        "--ma-long", 
        type=int, 
        help="Long MA window (e.g., 34) - required for conversion"
    )
    parser.add_argument(
        "--output", 
        help="Output filename (default: best_strategy_ma_X_Y.json)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Just validate existing best_strategy.json"
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        # Just validate existing file
        with open(args.backtest_file, 'r') as f:
            config = json.load(f)
        validate_live_format(config)
    else:
        # Convert backtest results
        if not args.ma_short or not args.ma_long:
            print("❌ ERROR: --ma-short and --ma-long required for conversion")
            sys.exit(1)
        
        config = convert_backtest_to_live_format(
            args.backtest_file,
            args.ma_short,
            args.ma_long,
            args.output
        )
        validate_live_format(config)
        
        print("\nTo deploy:")
        print(f"1. cp best_strategy.json best_strategy.json.backup_$(date +%Y%m%d)")
        print(f"2. cp {args.output or f'best_strategy_ma_{args.ma_short}_{args.ma_long}.json'} best_strategy.json")
        print("3. git add best_strategy.json && git commit -m 'Update MA parameters' && git push")


if __name__ == "__main__":
    main()