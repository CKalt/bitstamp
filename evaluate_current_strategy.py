#!/usr/bin/env python3
"""
Evaluate how current best_strategy.json would have performed over last N days
This tests the ACTUAL live configuration against recent history
"""
import json
import subprocess
import sys
from datetime import datetime, timedelta
import argparse


def create_evaluation_config(best_strategy, days_back):
    """Create config file from current best_strategy.json for backtesting"""
    
    # Extract the key parameters from best_strategy.json
    config = {
        # MA parameters - use exact values from live system
        "Short_Window": best_strategy.get("Short_Window", 6),
        "Long_Window": best_strategy.get("Long_Window", 34),
        
        # Trading parameters from live config
        "initial_balance": 10000,
        "fee_rate": 0.0012,
        "slippage_rate": 0.0005,
        
        # Use actual live settings
        "ma_separation_threshold": best_strategy.get("ma_separation_threshold", 0.3),
        "min_time_between_trades_minutes": best_strategy.get("min_time_between_trades_minutes", 30),
        "max_trades_per_day": best_strategy.get("max_trades_per_day", 5),
        "max_trades_per_hour": best_strategy.get("max_trades_per_hour", 2),
        
        # Pivot settings (disabled in current config)
        "enable_pivot_protection": best_strategy.get("enable_pivot_protection", False),
        "pivot_buffer": 100,  # Default since not in config
        "enable_trailing_pivots": False,
        
        # Other settings
        "enable_regime_detection": best_strategy.get("enable_regime_detection", False),
        "enable_adaptive_strategy": best_strategy.get("enable_adaptive_strategy", False),
        "strategy_type": best_strategy.get("strategy_type", "MA")
    }
    
    filename = f"eval_config_{days_back}days.json"
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    return filename


def run_evaluation(days_back):
    """Run backtest for last N days using current strategy"""
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n{'='*60}")
    print(f"Evaluating Current Strategy (MA {current_strategy['Short_Window']}/{current_strategy['Long_Window']})")
    print(f"Period: Last {days_back} days")
    print(f"From: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")
    
    # Create config
    config_file = create_evaluation_config(current_strategy, days_back)
    output_file = f"eval_current_{days_back}days.json"
    
    # Run backtest
    # Use virtual environment python
    cmd = [
        "env/bin/python", "src/bktst.py",
        "--data", "btcusd.log",
        "--config", config_file,
        "--start-date", start_date.strftime("%Y-%m-%d"),
        "--end-date", end_date.strftime("%Y-%m-%d"),
        "--save-results", output_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return None
    
    # Parse output
    print(result.stdout)
    
    # Load and display key metrics
    try:
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        print(f"\n📊 KEY METRICS FOR LAST {days_back} DAYS:")
        print(f"   Total Return: {results.get('total_return', 0) * 100:.2f}%")
        print(f"   Number of Trades: {results.get('num_trades', 0)}")
        print(f"   Final Equity: ${results.get('final_equity', 10000):.2f}")
        
        # Clean up temp config
        import os
        os.remove(config_file)
        
        return results
        
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return None


# Load current best_strategy.json
current_strategy = {
    "Average_Trades_Per_Day": 1.5161290322580645,
    "Bar_Size": "1H",
    "Final_Balance": 10877.413931682839,
    "Frequency": "1H",
    "Last_Signal_Action": "GO LONG",
    "Last_Signal_Timestamp": 1753099200,
    "Last_Trade_Timestamp": 1753102567,
    "Long_Window": 34,
    "Profit_Factor": 1.1236390721081366,
    "Sharpe_Ratio": 0.6001439204449514,
    "Short_Window": 6,
    "Strategy": "MA",
    "Total_Return": 8.774139316828386,
    "Total_Trades": 47.0,
    "auto_resume": True,
    "consecutive_loss_limit": 3,
    "daily_loss_limit": -2000,
    "do_live_trades": True,
    "emergency_loss_threshold": -10000,
    "emergency_override_enabled": False,
    "ignore_small_crosses": True,
    "log_signal_evaluation": True,
    "ma_separation_threshold": 0.3,
    "max_trades_per_day": 5,
    "max_trades_per_hour": 2,
    "min_breakout_distance_percent": 0.2,
    "min_time_between_trades_minutes": 30,
    "pause_after_limit_hours": 4,
    "pivot_buffer_multiplier": 1.5,
    "reduce_size_after_whipsaw": False,
    "require_confirmation_bars": 0,
    "use_dynamic_pivots": False,
    "verbose_logging": True,
    "volatility_lookback_hours": 24,
    "whipsaw_lookback_hours": 4,
    "whipsaw_size_reduction": 0.7,
    "enable_pivot_protection": False,
    "enable_regime_detection": False,
    "enable_adaptive_strategy": False,
    "strategy_type": "MA",
    "volatility_adjusted_pivots": False,
    "Last_Trade_Price": 118175
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate current best_strategy.json performance on recent data"
    )
    parser.add_argument(
        "--days",
        type=int,
        nargs='+',
        default=[30, 60],
        help="Days to look back (default: 30 60)"
    )
    
    args = parser.parse_args()
    
    print(f"Current Live Strategy Configuration:")
    print(f"MA: {current_strategy['Short_Window']}/{current_strategy['Long_Window']}")
    print(f"Original Sharpe Ratio: {current_strategy['Sharpe_Ratio']:.3f}")
    print(f"Original Total Return: {current_strategy['Total_Return']:.2f}%")
    print(f"Original creation date: Unknown (need to check when this was created)")
    
    results_summary = []
    
    for days in args.days:
        results = run_evaluation(days)
        if results:
            results_summary.append({
                'days': days,
                'return': results.get('total_return', 0) * 100,
                'trades': results.get('num_trades', 0),
                'sharpe': results.get('sharpe_ratio', 0)
            })
    
    # Print comparison
    if results_summary:
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"{'Period':<15} {'Return':>10} {'Trades':>10} {'Sharpe':>10}")
        print(f"{'-'*45}")
        print(f"{'Original':<15} {current_strategy['Total_Return']:>9.2f}% "
              f"{int(current_strategy['Total_Trades']):>10} "
              f"{current_strategy['Sharpe_Ratio']:>10.3f}")
        for r in results_summary:
            print(f"{'Last ' + str(r['days']) + ' days':<15} {r['return']:>9.2f}% "
                  f"{r['trades']:>10} {r['sharpe']:>10.3f}")
        
        print(f"\n💡 INSIGHTS:")
        for r in results_summary:
            if r['return'] < 0:
                print(f"⚠️  Strategy would have LOST {abs(r['return']):.2f}% in last {r['days']} days")
            else:
                print(f"✅ Strategy would have GAINED {r['return']:.2f}% in last {r['days']} days")
            
            if r['sharpe'] < current_strategy['Sharpe_Ratio'] * 0.5:
                print(f"⚠️  Sharpe ratio degraded significantly ({r['sharpe']:.3f} vs {current_strategy['Sharpe_Ratio']:.3f})")


if __name__ == "__main__":
    main()