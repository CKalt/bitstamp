#!/usr/bin/env python3
"""
Quick MA backtest without pandas dependency
Tests different MA periods on recent btcusd.log data
"""

import json
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

def parse_log_line(line: str) -> dict:
    """Parse a single line from btcusd.log"""
    try:
        data = json.loads(line.strip())
        if data.get('event') == 'trade' and 'data' in data:
            trade = data['data']
            return {
                'timestamp': trade.get('timestamp'),
                'price': float(trade.get('price', 0)),
                'amount': float(trade.get('amount', 0))
            }
        return None
    except:
        return None

def calculate_sma(prices: List[float], window: int) -> float:
    """Calculate simple moving average"""
    if len(prices) < window:
        return None
    return sum(prices[-window:]) / window

def backtest_ma_strategy(data: List[dict], short_window: int, long_window: int, 
                        initial_balance: float = 100000) -> dict:
    """Run backtest for MA crossover strategy"""
    position = "none"  # "long" or "none" 
    balance_usd = initial_balance
    balance_btc = 0
    trades = []
    
    prices = []
    
    for i, row in enumerate(data):
        if not row or 'price' not in row:
            continue
            
        price = float(row['price'])
        prices.append(price)
        
        if len(prices) < long_window:
            continue
            
        ma_short = calculate_sma(prices, short_window)
        ma_long = calculate_sma(prices, long_window)
        
        if ma_short is None or ma_long is None:
            continue
        
        # Trading logic
        if position == "none" and ma_short > ma_long:
            # Buy signal
            btc_amount = (balance_usd * 0.9988) / price  # 0.12% fee
            balance_btc = btc_amount
            balance_usd = 0
            position = "long"
            trades.append({
                "time": row.get('timestamp', i),
                "action": "buy",
                "price": price,
                "amount": btc_amount
            })
            
        elif position == "long" and ma_short < ma_long:
            # Sell signal
            usd_amount = balance_btc * price * 0.9988  # 0.12% fee
            balance_usd = usd_amount
            balance_btc = 0
            position = "none"
            trades.append({
                "time": row.get('timestamp', i),
                "action": "sell", 
                "price": price,
                "amount": balance_btc
            })
    
    # Final value
    if position == "long":
        final_value = balance_btc * prices[-1]
    else:
        final_value = balance_usd
        
    return {
        "short_window": short_window,
        "long_window": long_window,
        "final_value": final_value,
        "return_pct": ((final_value - initial_balance) / initial_balance) * 100,
        "num_trades": len(trades),
        "trades": trades
    }

def main():
    print("Reading btcusd.log...")
    
    # Read last 30 days of data
    data = []
    cutoff_time = datetime.now() - timedelta(days=30)
    
    with open("btcusd.log", "r") as f:
        for line in f:
            row = parse_log_line(line)
            if row and 'timestamp' in row:
                try:
                    timestamp = datetime.fromtimestamp(float(row['timestamp']))
                    if timestamp > cutoff_time:
                        data.append(row)
                except:
                    continue
    
    print(f"Loaded {len(data)} data points from last 30 days")
    
    # Test different MA combinations
    test_cases = [
        (3, 22),   # Very fast
        (6, 34),   # Current live
        (8, 40),   # Slightly slower
        (10, 46),  # Medium
        (12, 48),  # Recommended earlier
        (15, 60),  # Conservative
        (20, 80),  # Very conservative
        (5, 20),   # Fast
        (10, 30),  # Medium fast
        (15, 45),  # Medium slow
    ]
    
    results = []
    
    print("\nRunning backtests...")
    for short, long in test_cases:
        result = backtest_ma_strategy(data, short, long)
        results.append(result)
        print(f"MA {short}/{long}: Return: {result['return_pct']:.2f}%, Trades: {result['num_trades']}")
    
    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    
    print("\n=== TOP 5 STRATEGIES ===")
    for i, result in enumerate(results[:5]):
        print(f"{i+1}. MA {result['short_window']}/{result['long_window']}: "
              f"Return: {result['return_pct']:.2f}%, Trades: {result['num_trades']}")
    
    # Save best strategy
    best = results[0]
    best_strategy = {
        "Short_Window": best['short_window'],
        "Long_Window": best['long_window'],
        "strategy": "MovingAverage", 
        "do_live_trades": True,
        "estimated_return_30d": best['return_pct'],
        "estimated_trades_30d": best['num_trades']
    }
    
    with open("best_strategy_backtest.json", "w") as f:
        json.dump(best_strategy, f, indent=2)
    
    print(f"\nBest strategy saved to best_strategy_backtest.json")
    print(f"Recommended: MA {best['short_window']}/{best['long_window']}")

if __name__ == "__main__":
    main()