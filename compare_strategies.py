#!/usr/bin/env python3
"""
Compare different strategy configurations to find optimal parameters
"""

import json
import subprocess
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

def run_backtest_config(config: Dict, data_file: str, start_date: str = None, end_date: str = None) -> Dict:
    """Run a single backtest with given configuration"""
    # Save config to temporary file
    temp_config = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(temp_config, 'w') as f:
        json.dump(config, f)
    
    # Build command
    cmd = [
        'python', 'src/bktst.py',
        '--data', data_file,
        '--config', temp_config,
        '--save-results', f"{temp_config}.results.json"
    ]
    
    if start_date:
        cmd.extend(['--start-date', start_date])
    if end_date:
        cmd.extend(['--end-date', end_date])
    
    # Run backtest
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Load results
        with open(f"{temp_config}.results.json", 'r') as f:
            results = json.load(f)
            
        # Cleanup
        os.remove(temp_config)
        os.remove(f"{temp_config}.results.json")
        
        return results
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        # Cleanup on error
        if os.path.exists(temp_config):
            os.remove(temp_config)
        if os.path.exists(f"{temp_config}.results.json"):
            os.remove(f"{temp_config}.results.json")
        return None

def create_config_variations() -> List[Dict]:
    """Create different configuration variations to test"""
    
    # Base configuration
    base_config = {
        "initial_balance": 10000,
        "fee_rate": 0.0012,
        "slippage_rate": 0.0005,
        "enable_pivot_protection": True,
        "pivot_buffer": 100,
        "pivot_lookback_hours": 2,
        "enable_trailing_pivots": True,
        "pivot_profit_tiers": [
            {"threshold": 0.05, "protection_ratio": 0.70},
            {"threshold": 0.10, "protection_ratio": 0.80},
            {"threshold": 0.15, "protection_ratio": 0.85},
            {"threshold": 0.20, "protection_ratio": 0.90}
        ],
        "pivot_respect_technical_levels": True,
        "regime_lookback": 100,
        "regime_switch_threshold": 0.80,
        "min_strategy_switch_minutes": 120,
        "min_trade_gap_minutes": 15,
        "signal_confirmation_bars": 2,
        "whipsaw_threshold": 8.0,
        "short_window": 10,
        "long_window": 20
    }
    
    variations = []
    
    # 1. Current configuration (baseline)
    config = base_config.copy()
    config['name'] = "Current Settings"
    variations.append(config)
    
    # 2. No pivot protection
    config = base_config.copy()
    config['name'] = "No Pivot Protection"
    config['enable_pivot_protection'] = False
    variations.append(config)
    
    # 3. Wider pivot buffer
    config = base_config.copy()
    config['name'] = "Wide Buffer ($200)"
    config['pivot_buffer'] = 200
    variations.append(config)
    
    # 4. Tighter pivot buffer
    config = base_config.copy()
    config['name'] = "Tight Buffer ($50)"
    config['pivot_buffer'] = 50
    variations.append(config)
    
    # 5. No trailing pivots
    config = base_config.copy()
    config['name'] = "Static Pivots Only"
    config['enable_trailing_pivots'] = False
    variations.append(config)
    
    # 6. Faster MA
    config = base_config.copy()
    config['name'] = "Fast MA (5/15)"
    config['short_window'] = 5
    config['long_window'] = 15
    variations.append(config)
    
    # 7. Slower MA
    config = base_config.copy()
    config['name'] = "Slow MA (20/50)"
    config['short_window'] = 20
    config['long_window'] = 50
    variations.append(config)
    
    # 8. Lower regime threshold
    config = base_config.copy()
    config['name'] = "Easy Regime Switch (60%)"
    config['regime_switch_threshold'] = 0.60
    variations.append(config)
    
    # 9. Higher trade gap
    config = base_config.copy()
    config['name'] = "30min Trade Gap"
    config['min_trade_gap_minutes'] = 30
    variations.append(config)
    
    # 10. More conservative profit tiers
    config = base_config.copy()
    config['name'] = "Conservative Profit Lock"
    config['pivot_profit_tiers'] = [
        {"threshold": 0.03, "protection_ratio": 0.50},
        {"threshold": 0.05, "protection_ratio": 0.60},
        {"threshold": 0.10, "protection_ratio": 0.70},
        {"threshold": 0.15, "protection_ratio": 0.80}
    ]
    variations.append(config)
    
    return variations

def compare_results(results_list: List[Dict]) -> pd.DataFrame:
    """Create comparison dataframe from results"""
    
    comparison_data = []
    
    for result in results_list:
        if result:
            comparison_data.append({
                'Strategy': result['config_name'],
                'Total Return %': round(result['total_return_pct'], 2),
                'Sharpe Ratio': round(result['sharpe_ratio'], 2),
                'Max Drawdown %': round(result['max_drawdown_pct'], 2),
                'Win Rate %': round(result['win_rate'], 1),
                'Total Trades': result['total_trades'],
                'Pivot Trades': result['pivot_trades'],
                'Pivot Win Rate %': round(result.get('pivot_win_rate', 0), 1),
                'Total Fees': round(result['total_fees'], 2),
                'Profit Factor': round(result['profit_factor'], 2)
            })
    
    return pd.DataFrame(comparison_data)

def plot_comparison(df: pd.DataFrame):
    """Create visualization of strategy comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Strategy Comparison Results', fontsize=16)
    
    # 1. Returns vs Drawdown
    ax1 = axes[0, 0]
    ax1.scatter(df['Max Drawdown %'], df['Total Return %'])
    for idx, row in df.iterrows():
        ax1.annotate(row['Strategy'], (row['Max Drawdown %'], row['Total Return %']), 
                    fontsize=8, rotation=45)
    ax1.set_xlabel('Max Drawdown %')
    ax1.set_ylabel('Total Return %')
    ax1.set_title('Risk vs Return')
    ax1.grid(True, alpha=0.3)
    
    # 2. Sharpe Ratio comparison
    ax2 = axes[0, 1]
    df_sorted = df.sort_values('Sharpe Ratio', ascending=True)
    ax2.barh(df_sorted['Strategy'], df_sorted['Sharpe Ratio'])
    ax2.set_xlabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Returns')
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate vs Number of Trades
    ax3 = axes[1, 0]
    ax3.scatter(df['Total Trades'], df['Win Rate %'])
    for idx, row in df.iterrows():
        ax3.annotate(row['Strategy'][:10], (row['Total Trades'], row['Win Rate %']), 
                    fontsize=8)
    ax3.set_xlabel('Total Trades')
    ax3.set_ylabel('Win Rate %')
    ax3.set_title('Trade Frequency vs Success Rate')
    ax3.grid(True, alpha=0.3)
    
    # 4. Pivot Protection Effectiveness
    ax4 = axes[1, 1]
    pivot_data = df[df['Pivot Trades'] > 0]
    if not pivot_data.empty:
        ax4.bar(pivot_data['Strategy'], pivot_data['Pivot Win Rate %'])
        ax4.set_ylabel('Pivot Win Rate %')
        ax4.set_title('Pivot Protection Effectiveness')
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No Pivot Data', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparison chart saved as 'strategy_comparison.png'")

def main():
    """Run strategy comparison"""
    
    print("=== Strategy Configuration Comparison ===\n")
    
    # Check if data file exists
    data_file = "btcusd.log"
    if not os.path.exists(data_file):
        print(f"Error: Data file {data_file} not found")
        return
    
    # Get date range (default: last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    print(f"Testing period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Data file: {data_file}\n")
    
    # Create configuration variations
    variations = create_config_variations()
    print(f"Testing {len(variations)} strategy variations...\n")
    
    # Run backtests
    results = []
    for i, config in enumerate(variations):
        print(f"[{i+1}/{len(variations)}] Testing: {config['name']}...")
        result = run_backtest_config(
            config, 
            data_file,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        if result:
            result['config_name'] = config['name']
            results.append(result)
    
    # Create comparison
    print("\n=== RESULTS SUMMARY ===\n")
    df = compare_results(results)
    
    # Sort by Sharpe ratio
    df_sorted = df.sort_values('Sharpe Ratio', ascending=False)
    print(df_sorted.to_string(index=False))
    
    # Save detailed results
    with open('strategy_comparison_detailed.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to 'strategy_comparison_detailed.json'")
    
    # Create visualization
    try:
        plot_comparison(df)
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    # Find best configuration
    print("\n=== BEST CONFIGURATIONS ===\n")
    
    print("Highest Return:")
    best_return = df_sorted.iloc[0]
    print(f"  {best_return['Strategy']}: {best_return['Total Return %']}%")
    
    print("\nBest Risk-Adjusted (Sharpe):")
    best_sharpe = df_sorted.iloc[0]
    print(f"  {best_sharpe['Strategy']}: {best_sharpe['Sharpe Ratio']}")
    
    print("\nLowest Drawdown:")
    best_dd = df.sort_values('Max Drawdown %', ascending=False).iloc[0]
    print(f"  {best_dd['Strategy']}: {best_dd['Max Drawdown %']}%")

if __name__ == "__main__":
    main()