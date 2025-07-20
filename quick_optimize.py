#!/usr/bin/env python3
"""
Quick parameter optimization focusing on 90-day period
"""

import os
import sys
import json
import yaml
import subprocess
from datetime import datetime, timedelta

def test_configurations():
    """Test a few key configurations"""
    
    # Base configuration template
    base_config = """
name: "{name}"
description: "{description}"
data_source: "btcusd.log"
initial_btc: 0.0
initial_usd: 100000.0
always_in_market: true
fee_percentage: 0.0012
slippage_bps: 0.0
max_trades_per_day: 5
max_trades_per_hour: 3
min_trade_gap_minutes: 15
min_btc_trade_size: 0.00000001
enable_pivot_protection: true
enable_trailing_stops: true
emergency_exit_loss: -2000.0
pivot_buffer: 100.0
strategy:
  name: "Adaptive Multi-Strategy"
  type: "adaptive"
  parameters: {{}}
output_dir: "backtest_results"
save_trades: true
save_signals: true
save_equity_curve: true
generate_plots: false
"""

    configs = [
        # 1. More aggressive ranging strategy
        {
            "name": "aggressive_ranging",
            "config": base_config + """
regime_detection:
  lookback_bars: 100
  whipsaw_threshold: 0.65
  trend_strength_threshold: 0.3
  volatility_window: 50
  confidence_threshold: 0.6
trending_strategy:
  short_window: 10
  long_window: 30
  confirmation_bars: 2
ranging_strategy:
  bb_window: 15
  bb_std_dev: 1.5
  rsi_window: 14
  rsi_oversold: 35
  rsi_overbought: 65
  exit_at_opposite_band: true
volatile_strategy:
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  breakout_threshold: 100
"""
        },
        # 2. Conservative approach
        {
            "name": "conservative",
            "config": base_config + """
regime_detection:
  lookback_bars: 150
  whipsaw_threshold: 0.75
  trend_strength_threshold: 0.4
  volatility_window: 75
  confidence_threshold: 0.7
trending_strategy:
  short_window: 15
  long_window: 40
  confirmation_bars: 3
ranging_strategy:
  bb_window: 25
  bb_std_dev: 2.5
  rsi_window: 14
  rsi_oversold: 25
  rsi_overbought: 75
  exit_at_opposite_band: true
volatile_strategy:
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  breakout_threshold: 150
"""
        },
        # 3. Optimized for recent market
        {
            "name": "optimized_recent",
            "config": base_config + """
regime_detection:
  lookback_bars: 80
  whipsaw_threshold: 0.60
  trend_strength_threshold: 0.25
  volatility_window: 40
  confidence_threshold: 0.55
trending_strategy:
  short_window: 8
  long_window: 25
  confirmation_bars: 2
ranging_strategy:
  bb_window: 18
  bb_std_dev: 1.8
  rsi_window: 14
  rsi_oversold: 32
  rsi_overbought: 68
  exit_at_opposite_band: true
volatile_strategy:
  macd_fast: 10
  macd_slow: 24
  macd_signal: 8
  breakout_threshold: 80
"""
        }
    ]
    
    # Test each configuration
    results = []
    
    print("Testing configurations on 90-day period...\n")
    
    for cfg in configs:
        name = cfg["name"]
        config_content = cfg["config"].format(
            name=f"Test {name}",
            description=f"Testing {name} configuration"
        )
        
        # Save config
        config_file = f"config/strategies/test_{name}.yaml"
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        # Run backtest
        print(f"Testing {name}...", end='', flush=True)
        output_file = f"test_{name}_90days.json"
        
        cmd = [
            "./run_backtest.sh",
            "--start", "2025-04-19",
            "--end", "2025-07-18",
            "--config", config_file,
            "--output", output_file,
            "--quiet"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Load results
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            result = {
                "name": name,
                "file": output_file,
                "return": data.get("total_return", 0) * 100,
                "sharpe": data.get("metrics", {}).get("risk", {}).get("sharpe_ratio", 0),
                "trades": data.get("num_trades", 0),
                "win_rate": data.get("metrics", {}).get("win_loss_analysis", {}).get("win_rate", 0) * 100,
                "max_dd": data.get("metrics", {}).get("risk", {}).get("max_drawdown_pct", 0)
            }
            results.append(result)
            print(f" Return: {result['return']:.2f}%, Sharpe: {result['sharpe']:.3f}")
            
        except Exception as e:
            print(f" Failed: {e}")
        
        # Clean up config
        os.remove(config_file)
    
    # Also test the default
    print(f"\nTesting default configuration...", end='', flush=True)
    cmd = [
        "./run_backtest.sh",
        "--start", "2025-04-19",
        "--end", "2025-07-18",
        "--output", "test_default_90days.json",
        "--quiet"
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open("test_default_90days.json", 'r') as f:
            data = json.load(f)
        
        result = {
            "name": "default",
            "file": "test_default_90days.json",
            "return": data.get("total_return", 0) * 100,
            "sharpe": data.get("metrics", {}).get("risk", {}).get("sharpe_ratio", 0),
            "trades": data.get("num_trades", 0),
            "win_rate": data.get("metrics", {}).get("win_loss_analysis", {}).get("win_rate", 0) * 100,
            "max_dd": data.get("metrics", {}).get("risk", {}).get("max_drawdown_pct", 0)
        }
        results.append(result)
        print(f" Return: {result['return']:.2f}%, Sharpe: {result['sharpe']:.3f}")
    except Exception as e:
        print(f" Failed: {e}")
    
    # Show results
    print("\n\n=== RESULTS SUMMARY (90-day backtest) ===\n")
    
    # Sort by return
    results.sort(key=lambda x: x['return'], reverse=True)
    
    print("Configuration     Return    Sharpe   Trades  WinRate   MaxDD")
    print("-" * 65)
    for r in results:
        print(f"{r['name']:15} {r['return']:7.2f}%  {r['sharpe']:7.3f}  {r['trades']:6}  {r['win_rate']:6.1f}%  {r['max_dd']:6.2f}%")
    
    # Find best
    best = max(results, key=lambda x: x['sharpe'] * x['return'] / 100)
    
    print(f"\n\n=== RECOMMENDATION ===")
    print(f"\nBest configuration: {best['name']}")
    print(f"- Return: {best['return']:.2f}%")
    print(f"- Sharpe Ratio: {best['sharpe']:.3f}")
    print(f"- Win Rate: {best['win_rate']:.1f}%")
    print(f"- Results file: {best['file']}")
    
    print(f"\nTo deploy to live trading:")
    print(f"1. Review full results:")
    print(f"   cat {best['file']} | jq '.metrics'")
    print(f"2. Deploy with backup:")
    print(f"   python src/backtesting/deploy_strategy.py {best['file']} --backup")
    print(f"3. Restart your auto-trade system")

if __name__ == "__main__":
    test_configurations()