#!/usr/bin/env python3
"""
Compare live trading logs with backtest results
Verifies that trades happen at the exact same times
"""
import json
import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from data.loader import parse_log_file


def load_live_trades(log_date):
    """Load trades from live trading logs"""
    log_file = f"logs/backtest_comparison/live_trading_{log_date}.jsonl"
    
    if not os.path.exists(log_file):
        print(f"❌ Live trading log not found: {log_file}")
        return []
    
    trades = []
    hourly_bars = []
    
    with open(log_file, "r") as f:
        for line in f:
            event = json.loads(line)
            if event["event_type"] == "TRADE_EXECUTION":
                trades.append({
                    "time": event["hourly_bar_time"],
                    "exact_time": event["exact_trigger_time"],
                    "type": event["trade"]["type"],
                    "price": event["trade"]["price"],
                    "ma_short": event["indicators_at_trade"]["ma_short"],
                    "ma_long": event["indicators_at_trade"]["ma_long"],
                    "hash": event["verification_hash"]
                })
                
    # Also load hourly bars
    hourly_file = f"logs/backtest_comparison/hourly_bars_{log_date}.jsonl"
    if os.path.exists(hourly_file):
        with open(hourly_file, "r") as f:
            for line in f:
                event = json.loads(line)
                if event["event_type"] == "HOURLY_BAR":
                    hourly_bars.append({
                        "time": event["bar_time"],
                        "close": event["ohlcv"]["close"],
                        "ma_short": event["indicators"]["ma_short_value"],
                        "ma_long": event["indicators"]["ma_long_value"]
                    })
    
    return trades, hourly_bars


def run_backtest_for_period(start_date, end_date, ma_short, ma_long):
    """Run backtest for specific period and parameters"""
    print(f"Running backtest for {start_date} to {end_date} with MA {ma_short}/{ma_long}")
    
    # Load data
    df = parse_log_file('btcusd.log', 
                       start_date=datetime.strptime(start_date, "%Y-%m-%d"),
                       end_date=datetime.strptime(end_date, "%Y-%m-%d"))
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('datetime', inplace=True)
    
    # Resample to hourly
    df_hourly = df.resample('1H').agg({
        'price': ['first', 'max', 'min', 'last'],
        'amount': 'sum'
    })
    df_hourly.columns = ['open', 'high', 'low', 'close', 'volume']
    df_hourly = df_hourly.dropna()
    
    # Calculate MAs
    df_hourly['MA_short'] = df_hourly['close'].rolling(window=ma_short).mean()
    df_hourly['MA_long'] = df_hourly['close'].rolling(window=ma_long).mean()
    
    # Generate signals
    df_hourly['signal'] = 0
    df_hourly.loc[df_hourly['MA_short'] > df_hourly['MA_long'], 'signal'] = 1
    df_hourly.loc[df_hourly['MA_short'] < df_hourly['MA_long'], 'signal'] = -1
    
    # Find trades
    trades = []
    position = 0
    
    for i in range(1, len(df_hourly)):
        prev_signal = df_hourly.iloc[i-1]['signal']
        curr_signal = df_hourly.iloc[i]['signal']
        
        if prev_signal != curr_signal and curr_signal != 0:
            bar_time = df_hourly.index[i]
            
            if curr_signal == 1 and position <= 0:
                # BUY signal
                trades.append({
                    "time": bar_time.isoformat(),
                    "type": "BUY",
                    "price": df_hourly.iloc[i]['close'],
                    "ma_short": df_hourly.iloc[i]['MA_short'],
                    "ma_long": df_hourly.iloc[i]['MA_long']
                })
                position = 1
            elif curr_signal == -1 and position >= 0:
                # SELL signal
                trades.append({
                    "time": bar_time.isoformat(),
                    "type": "SELL",
                    "price": df_hourly.iloc[i]['close'],
                    "ma_short": df_hourly.iloc[i]['MA_short'],
                    "ma_long": df_hourly.iloc[i]['MA_long']
                })
                position = -1
    
    return trades, df_hourly


def compare_trades(live_trades, backtest_trades):
    """Compare live trades with backtest trades"""
    print("\n" + "="*80)
    print("TRADE COMPARISON")
    print("="*80)
    
    # Convert to sets for comparison
    live_times = {t["time"] for t in live_trades}
    backtest_times = {t["time"] for t in backtest_trades}
    
    # Find matches and mismatches
    matched = live_times & backtest_times
    live_only = live_times - backtest_times
    backtest_only = backtest_times - live_times
    
    print(f"\n📊 Summary:")
    print(f"   Live trades: {len(live_trades)}")
    print(f"   Backtest trades: {len(backtest_trades)}")
    print(f"   Matched trades: {len(matched)}")
    print(f"   Live-only trades: {len(live_only)}")
    print(f"   Backtest-only trades: {len(backtest_only)}")
    
    # Show matched trades
    if matched:
        print(f"\n✅ Matched Trades ({len(matched)}):")
        for trade_time in sorted(matched):
            live_trade = next(t for t in live_trades if t["time"] == trade_time)
            backtest_trade = next(t for t in backtest_trades if t["time"] == trade_time)
            
            price_diff = abs(live_trade["price"] - backtest_trade["price"])
            ma_short_diff = abs(live_trade["ma_short"] - backtest_trade["ma_short"])
            ma_long_diff = abs(live_trade["ma_long"] - backtest_trade["ma_long"])
            
            print(f"   {trade_time} - {live_trade['type']}")
            print(f"      Price: Live=${live_trade['price']:.2f}, Backtest=${backtest_trade['price']:.2f} (diff=${price_diff:.2f})")
            print(f"      MA Short: Live={live_trade['ma_short']:.2f}, Backtest={backtest_trade['ma_short']:.2f} (diff={ma_short_diff:.2f})")
            print(f"      MA Long: Live={live_trade['ma_long']:.2f}, Backtest={backtest_trade['ma_long']:.2f} (diff={ma_long_diff:.2f})")
    
    # Show mismatches
    if live_only:
        print(f"\n❌ Live-Only Trades (not in backtest):")
        for trade_time in sorted(live_only):
            live_trade = next(t for t in live_trades if t["time"] == trade_time)
            print(f"   {trade_time} - {live_trade['type']} @ ${live_trade['price']:.2f}")
            print(f"      MA: {live_trade['ma_short']:.2f} / {live_trade['ma_long']:.2f}")
    
    if backtest_only:
        print(f"\n❌ Backtest-Only Trades (missed by live):")
        for trade_time in sorted(backtest_only):
            backtest_trade = next(t for t in backtest_trades if t["time"] == trade_time)
            print(f"   {trade_time} - {backtest_trade['type']} @ ${backtest_trade['price']:.2f}")
            print(f"      MA: {backtest_trade['ma_short']:.2f} / {backtest_trade['ma_long']:.2f}")
    
    # Calculate accuracy
    if len(backtest_trades) > 0:
        accuracy = len(matched) / len(backtest_trades) * 100
        print(f"\n📈 Accuracy: {accuracy:.1f}% of backtest trades were matched")
    
    return {
        "matched": len(matched),
        "live_only": len(live_only),
        "backtest_only": len(backtest_only),
        "accuracy": accuracy if len(backtest_trades) > 0 else 0
    }


def main():
    """Main comparison function"""
    # Get date from command line or use today
    if len(sys.argv) > 1:
        log_date = sys.argv[1]
    else:
        log_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Comparing live trading vs backtest for: {log_date}")
    
    # Load live trades
    live_trades, hourly_bars = load_live_trades(log_date)
    
    if not live_trades and not hourly_bars:
        print("No live trading data found for this date")
        return
    
    # Get MA parameters from first hourly bar or trade
    if hourly_bars:
        # Estimate MA parameters from the data
        # For now, assume MA 3/22 as that's what we're testing
        ma_short = 3
        ma_long = 22
    else:
        print("No hourly bar data found")
        return
    
    # Run backtest for the same period
    start_date = log_date
    end_date = (datetime.strptime(log_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    backtest_trades, backtest_data = run_backtest_for_period(start_date, end_date, ma_short, ma_long)
    
    # Compare results
    comparison = compare_trades(live_trades, backtest_trades)
    
    # Save comparison report
    report = {
        "date": log_date,
        "ma_parameters": {"short": ma_short, "long": ma_long},
        "live_trades": live_trades,
        "backtest_trades": backtest_trades,
        "comparison": comparison,
        "generated_at": datetime.now().isoformat()
    }
    
    report_file = f"logs/backtest_comparison/comparison_{log_date}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Comparison report saved to: {report_file}")


if __name__ == "__main__":
    main()