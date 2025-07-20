#!/usr/bin/env python3
"""
Parameter optimization script for backtesting
Tests multiple parameter configurations and finds the best performing ones
"""

import os
import sys
import json
import yaml
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple

def create_config_variations():
    """Create different parameter configurations to test"""
    
    base_config = {
        "name": "Optimization Test",
        "description": "Testing parameter variations",
        "data_source": "btcusd.log",
        "initial_btc": 0.0,
        "initial_usd": 100000.0,
        "always_in_market": True,
        "fee_percentage": 0.0012,
        "slippage_bps": 0.0,
        "max_trades_per_day": 5,
        "max_trades_per_hour": 3,
        "min_trade_gap_minutes": 15,
        "min_btc_trade_size": 0.00000001,
        "enable_pivot_protection": True,
        "enable_trailing_stops": True,
        "emergency_exit_loss": -2000.0,
        "pivot_buffer": 100.0,
        "strategy": {
            "name": "Adaptive Multi-Strategy",
            "type": "adaptive",
            "parameters": {}
        },
        "output_dir": "backtest_results",
        "save_trades": True,
        "save_signals": True,
        "save_equity_curve": True,
        "generate_plots": False
    }
    
    # Define parameter variations to test
    variations = []
    
    # 1. Conservative (fewer trades, stronger signals)
    conservative = base_config.copy()
    conservative.update({
        "name": "Conservative",
        "regime_detection": {
            "lookback_bars": 150,
            "whipsaw_threshold": 0.75,
            "trend_strength_threshold": 0.4,
            "volatility_window": 75,
            "confidence_threshold": 0.7
        },
        "trending_strategy": {
            "short_window": 15,
            "long_window": 40,
            "confirmation_bars": 3
        },
        "ranging_strategy": {
            "bb_window": 25,
            "bb_std_dev": 2.5,
            "rsi_window": 14,
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "exit_at_opposite_band": True
        },
        "volatile_strategy": {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "breakout_threshold": 150
        }
    })
    variations.append(("conservative", conservative))
    
    # 2. Aggressive (more trades, quicker signals)
    aggressive = base_config.copy()
    aggressive.update({
        "name": "Aggressive",
        "regime_detection": {
            "lookback_bars": 50,
            "whipsaw_threshold": 0.55,
            "trend_strength_threshold": 0.2,
            "volatility_window": 30,
            "confidence_threshold": 0.5
        },
        "trending_strategy": {
            "short_window": 5,
            "long_window": 20,
            "confirmation_bars": 1
        },
        "ranging_strategy": {
            "bb_window": 15,
            "bb_std_dev": 1.5,
            "rsi_window": 14,
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "exit_at_opposite_band": True
        },
        "volatile_strategy": {
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 7,
            "breakout_threshold": 50
        }
    })
    variations.append(("aggressive", aggressive))
    
    # 3. Balanced (default with minor tweaks)
    balanced = base_config.copy()
    balanced.update({
        "name": "Balanced",
        "regime_detection": {
            "lookback_bars": 100,
            "whipsaw_threshold": 0.65,
            "trend_strength_threshold": 0.3,
            "volatility_window": 50,
            "confidence_threshold": 0.6
        },
        "trending_strategy": {
            "short_window": 10,
            "long_window": 30,
            "confirmation_bars": 2
        },
        "ranging_strategy": {
            "bb_window": 20,
            "bb_std_dev": 2.0,
            "rsi_window": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "exit_at_opposite_band": True
        },
        "volatile_strategy": {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "breakout_threshold": 100
        }
    })
    variations.append(("balanced", balanced))
    
    # 4. Trend-focused
    trend_focused = base_config.copy()
    trend_focused.update({
        "name": "Trend Focused",
        "regime_detection": {
            "lookback_bars": 100,
            "whipsaw_threshold": 0.5,  # Easier to detect trends
            "trend_strength_threshold": 0.2,  # Lower threshold
            "volatility_window": 50,
            "confidence_threshold": 0.5
        },
        "trending_strategy": {
            "short_window": 8,
            "long_window": 25,
            "confirmation_bars": 2
        },
        "ranging_strategy": {
            "bb_window": 30,
            "bb_std_dev": 3.0,  # Wider bands, fewer signals
            "rsi_window": 14,
            "rsi_oversold": 20,
            "rsi_overbought": 80,
            "exit_at_opposite_band": True
        },
        "volatile_strategy": {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "breakout_threshold": 100
        }
    })
    variations.append(("trend_focused", trend_focused))
    
    # 5. Range-focused
    range_focused = base_config.copy()
    range_focused.update({
        "name": "Range Focused",
        "regime_detection": {
            "lookback_bars": 100,
            "whipsaw_threshold": 0.8,  # Harder to switch regimes
            "trend_strength_threshold": 0.5,  # Higher threshold
            "volatility_window": 50,
            "confidence_threshold": 0.7
        },
        "trending_strategy": {
            "short_window": 20,
            "long_window": 50,  # Slower signals
            "confirmation_bars": 4
        },
        "ranging_strategy": {
            "bb_window": 18,
            "bb_std_dev": 1.8,  # Tighter bands
            "rsi_window": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "exit_at_opposite_band": True
        },
        "volatile_strategy": {
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "breakout_threshold": 150
        }
    })
    variations.append(("range_focused", range_focused))
    
    return variations

def run_backtest(config_name: str, config: Dict, period: str = "90days") -> Dict:
    """Run a single backtest with given configuration"""
    
    # Save config to temporary file
    config_file = f"config/strategies/temp_{config_name}.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    # Determine date range based on period
    end_date = datetime.now().strftime("%Y-%m-%d")
    if period == "30days":
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    elif period == "90days":
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    elif period == "180days":
        start_date = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
    else:  # 1 year
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    # Run backtest
    output_file = f"optimization_{config_name}_{period}.json"
    cmd = [
        "python3", "src/backtesting/run_backtest.py",
        "--config", config_file,
        "--start-date", start_date,
        "--end-date", end_date,
        "--output-file", output_file,
        "--quiet"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Load results
        with open(output_file, 'r') as f:
            results = json.load(f)
        
        # Clean up
        os.remove(config_file)
        
        return {
            "config_name": config_name,
            "period": period,
            "total_return": results.get("total_return", 0),
            "sharpe_ratio": results.get("metrics", {}).get("risk", {}).get("sharpe_ratio", 0),
            "win_rate": results.get("metrics", {}).get("win_loss_analysis", {}).get("win_rate", 0),
            "num_trades": results.get("num_trades", 0),
            "max_drawdown": results.get("metrics", {}).get("risk", {}).get("max_drawdown_pct", 0),
            "profit_factor": results.get("metrics", {}).get("win_loss_analysis", {}).get("profit_factor", 0),
            "output_file": output_file
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running backtest for {config_name}: {e.stderr}")
        # Clean up
        if os.path.exists(config_file):
            os.remove(config_file)
        return None

def main():
    print("=== Parameter Optimization for Trading Strategy ===\n")
    
    # Activate virtual environment
    os.system("source env/bin/activate")
    
    # Create config variations
    variations = create_config_variations()
    
    # Test periods
    periods = ["30days", "90days", "180days"]
    
    # Run backtests
    all_results = []
    
    print(f"Testing {len(variations)} configurations over {len(periods)} time periods...")
    print("This may take several minutes...\n")
    
    for period in periods:
        print(f"\n--- Testing {period} period ---")
        for name, config in variations:
            print(f"  Running {name} configuration...", end='', flush=True)
            result = run_backtest(name, config, period)
            if result:
                all_results.append(result)
                print(f" Return: {result['total_return']*100:.2f}%, Sharpe: {result['sharpe_ratio']:.3f}")
            else:
                print(" Failed!")
    
    # Sort results by different metrics
    print("\n\n=== OPTIMIZATION RESULTS ===\n")
    
    # Best by Sharpe Ratio
    print("Top 5 by Sharpe Ratio:")
    sorted_by_sharpe = sorted([r for r in all_results if r], 
                             key=lambda x: x['sharpe_ratio'], reverse=True)[:5]
    for i, r in enumerate(sorted_by_sharpe, 1):
        print(f"{i}. {r['config_name']} ({r['period']}): "
              f"Sharpe={r['sharpe_ratio']:.3f}, "
              f"Return={r['total_return']*100:.2f}%, "
              f"Trades={r['num_trades']}")
    
    # Best by Total Return
    print("\nTop 5 by Total Return:")
    sorted_by_return = sorted([r for r in all_results if r], 
                             key=lambda x: x['total_return'], reverse=True)[:5]
    for i, r in enumerate(sorted_by_return, 1):
        print(f"{i}. {r['config_name']} ({r['period']}): "
              f"Return={r['total_return']*100:.2f}%, "
              f"Sharpe={r['sharpe_ratio']:.3f}, "
              f"MaxDD={r['max_drawdown']:.2f}%")
    
    # Best balanced (high Sharpe + good returns)
    print("\nBest Balanced (Sharpe * Return score):")
    sorted_by_balanced = sorted([r for r in all_results if r], 
                               key=lambda x: x['sharpe_ratio'] * x['total_return'], 
                               reverse=True)[:5]
    for i, r in enumerate(sorted_by_balanced, 1):
        score = r['sharpe_ratio'] * r['total_return']
        print(f"{i}. {r['config_name']} ({r['period']}): "
              f"Score={score:.3f}, "
              f"Sharpe={r['sharpe_ratio']:.3f}, "
              f"Return={r['total_return']*100:.2f}%")
    
    # Recommendation
    print("\n=== RECOMMENDATION ===")
    best = sorted_by_balanced[0] if sorted_by_balanced else None
    if best:
        print(f"\nBest overall configuration: {best['config_name']}")
        print(f"Results file: {best['output_file']}")
        print(f"\nTo deploy this configuration:")
        print(f"1. Review: cat {best['output_file']} | jq '.metrics'")
        print(f"2. Deploy: python src/backtesting/deploy_strategy.py {best['output_file']} --backup")
        print(f"3. Restart your auto-trade system to use the new parameters")
    
    # Save summary
    summary_file = "optimization_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
            "best_sharpe": sorted_by_sharpe[0] if sorted_by_sharpe else None,
            "best_return": sorted_by_return[0] if sorted_by_return else None,
            "best_balanced": best
        }, f, indent=2)
    print(f"\nFull results saved to: {summary_file}")

if __name__ == "__main__":
    main()