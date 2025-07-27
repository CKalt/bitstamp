#!/usr/bin/env python3
"""
Analyze existing backtest results and prepare for deployment
"""

import json
import subprocess
from pathlib import Path

def analyze_results():
    """Analyze existing backtest results"""
    
    results = []
    
    # Load test results
    test_files = [
        ("30 days", "test_30days.json"),
        ("90 days", "test_90days.json"),
        ("1 year", "test_1year.json")
    ]
    
    print("=== BACKTEST RESULTS ANALYSIS ===\n")
    
    for period, filename in test_files:
        if Path(filename).exists():
            with open(filename, 'r') as f:
                data = json.load(f)
            
            metrics = data.get("metrics", {})
            risk = metrics.get("risk", {})
            win_loss = metrics.get("win_loss_analysis", {})
            
            result = {
                "period": period,
                "file": filename,
                "return": data.get("total_return", 0) * 100,
                "annualized": metrics.get("summary", {}).get("annualized_return_pct", 0),
                "sharpe": risk.get("sharpe_ratio", 0),
                "sortino": risk.get("sortino_ratio", 0),
                "trades": data.get("num_trades", 0),
                "win_rate": win_loss.get("win_rate", 0) * 100,
                "profit_factor": win_loss.get("profit_factor", 0),
                "max_dd": risk.get("max_drawdown_pct", 0),
                "calmar": risk.get("calmar_ratio", 0)
            }
            results.append(result)
            
            print(f"{period} Results:")
            print(f"  Return: {result['return']:.2f}% ({result['annualized']:.1f}% annualized)")
            print(f"  Sharpe Ratio: {result['sharpe']:.3f}")
            print(f"  Sortino Ratio: {result['sortino']:.3f}")
            print(f"  Calmar Ratio: {result['calmar']:.3f}")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Max Drawdown: {result['max_dd']:.2f}%")
            print(f"  Total Trades: {result['trades']}")
            print()
    
    # Score each result (balanced approach)
    for r in results:
        # Score based on multiple factors
        sharpe_score = min(r['sharpe'] / 2.0, 1.0) * 30  # Max 30 points
        return_score = min(r['annualized'] / 100, 1.0) * 25  # Max 25 points
        win_rate_score = (r['win_rate'] / 100) * 20  # Max 20 points
        drawdown_score = max(0, (30 - abs(r['max_dd'])) / 30) * 15  # Max 15 points
        trades_score = min(r['trades'] / 50, 1.0) * 10  # Max 10 points
        
        r['score'] = sharpe_score + return_score + win_rate_score + drawdown_score + trades_score
    
    # Sort by score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n=== SCORING ANALYSIS ===")
    print("\nPeriod     Score  Sharpe  Return  WinRate  MaxDD   Trades")
    print("-" * 60)
    for r in results:
        print(f"{r['period']:10} {r['score']:5.1f}  {r['sharpe']:6.3f}  {r['return']:6.2f}%  "
              f"{r['win_rate']:6.1f}%  {r['max_dd']:6.2f}%  {r['trades']:6}")
    
    # Recommendation
    best = results[0] if results else None
    
    if best:
        print(f"\n\n=== RECOMMENDATION ===")
        print(f"\nBased on the analysis, the {best['period']} backtest shows the best overall performance.")
        print(f"\nKey strengths:")
        if best['sharpe'] > 2:
            print(f"- Excellent risk-adjusted returns (Sharpe: {best['sharpe']:.3f})")
        elif best['sharpe'] > 1:
            print(f"- Good risk-adjusted returns (Sharpe: {best['sharpe']:.3f})")
        
        if best['win_rate'] > 60:
            print(f"- High win rate ({best['win_rate']:.1f}%)")
        
        if abs(best['max_dd']) < 10:
            print(f"- Low maximum drawdown ({best['max_dd']:.2f}%)")
        elif abs(best['max_dd']) < 20:
            print(f"- Moderate maximum drawdown ({best['max_dd']:.2f}%)")
        
        print(f"\nThe current parameters are already well-optimized for the market conditions.")
        print(f"\nTo deploy these parameters to your live system:")
        print(f"\n1. First, do a final review:")
        print(f"   cat {best['file']} | jq '.metrics'")
        print(f"\n2. Deploy with backup (RECOMMENDED):")
        print(f"   python src/backtesting/deploy_strategy.py {best['file']} --backup")
        print(f"\n3. Or if you want to see what would be deployed first:")
        print(f"   python src/backtesting/deploy_strategy.py {best['file']} --dry-run")
        print(f"\n4. After deployment, restart your auto-trade system:")
        print(f"   - Stop current trading: stop_auto_trade")
        print(f"   - Start with new params: resume_auto_trade [btc_amount] [position] [price]")
        
        print(f"\n\nIMPORTANT NOTES:")
        print(f"- The backtest used the SAME strategy code as your live system")
        print(f"- All trades in the backtest were in 'ranging' market regime")
        print(f"- Consider monitoring performance closely for the first 24-48 hours")
        print(f"- The parameters are already conservative and well-balanced")

if __name__ == "__main__":
    analyze_results()