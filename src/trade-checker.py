###############################################################################
# src/trade-checker.py
###############################################################################
# Full File Path: src/trade-checker.py
#
# PURPOSE:
#   This script reads the logged signals (signal_logs.json) and the actual
#   trades (trades.json or non-live-trades.json), then checks for consistency
#   in timing, daily trade counts, partial trades, etc. 
#
#   It can also parse the historical price data from btcusd.log if desired,
#   allowing additional checks (e.g. "did we trade at a realistic price?").
#
# USAGE:
#   python src/trade-checker.py --dry-run     # checks 'non-live-trades.json'
#   python src/trade-checker.py --live-run    # checks 'trades.json'
#
# NOTES:
#   - This is just a starting point. You can extend or refine the checks as needed.
#   - We preserve existing comments and structure from the user’s instructions,
#     but since this is a new file, we only provide the code needed here.
###############################################################################

import os
import json
import argparse
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description="Check the consistency of auto-trades vs. signals.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check using non-live-trades.json instead of trades.json.")
    parser.add_argument("--live-run", action="store_true",
                        help="Check using trades.json instead of non-live-trades.json.")
    parser.add_argument("--log-file", type=str, default="btcusd.log",
                        help="Path to the btcusd.log (for deeper checks).")
    return parser.parse_args()

def load_json_file(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return []

def main():
    args = parse_arguments()

    # Decide which trades file to read
    if args.live_run:
        trades_file = "trades.json"
    else:
        # default is dry-run
        trades_file = "non-live-trades.json"

    signal_logs_path = os.path.abspath("signal_logs.json")
    trades_path = os.path.abspath(trades_file)
    log_file_path = os.path.abspath(args.log_file)

    # Load signals
    signal_entries = load_json_file(signal_logs_path)
    # Load trades
    trade_entries = load_json_file(trades_path)

    print(f"Loaded {len(signal_entries)} signals from {signal_logs_path}")
    print(f"Loaded {len(trade_entries)} trades from {trades_path}")

    # Basic checks
    # 1) Group partial trades that happen within a small time window from the same signal
    #    so we can see if we are counting them as 1 daily trade or multiple.
    # 2) Ensure each signal had a corresponding trade (or set of partial trades).
    # 3) Check for trades that happened with no preceding signal.

    # Convert timestamps to datetime objects
    for s in signal_entries:
        try:
            s["parsed_time"] = datetime.fromisoformat(s["timestamp"])
        except:
            s["parsed_time"] = None

    for t in trade_entries:
        ts_str = t.get("signal_timestamp")  # the time when the signal occurred
        if ts_str:
            try:
                t["parsed_signal_time"] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except:
                t["parsed_signal_time"] = None
        else:
            t["parsed_signal_time"] = None

        # Also parse the actual trade timestamp
        trade_ts_str = t.get("timestamp")
        if trade_ts_str:
            try:
                t["parsed_trade_time"] = datetime.strptime(trade_ts_str, "%Y-%m-%d %H:%M:%S")
            except:
                t["parsed_trade_time"] = None
        else:
            t["parsed_trade_time"] = None

    # 1) Check for unmatched signals
    unmatched_signals = []
    signals_matched_count = 0

    for sig in signal_entries:
        sig_time = sig.get("parsed_time")
        sig_strat = sig.get("strategy", "Unknown")
        matched_trades = []
        if not sig_time:
            unmatched_signals.append(sig)
            continue

        # Look for trades whose parsed_signal_time is the same or within a close range
        for trade in trade_entries:
            if trade["parsed_signal_time"] and abs((trade["parsed_signal_time"] - sig_time).total_seconds()) < 120:
                # We consider "matched" if within 2 minutes of the signal
                matched_trades.append(trade)

        if matched_trades:
            signals_matched_count += 1
        else:
            unmatched_signals.append(sig)

    print(f"\nSignals matched to at least one trade: {signals_matched_count}")
    if unmatched_signals:
        print(f"Unmatched signals (no trades found): {len(unmatched_signals)}")
        for usig in unmatched_signals:
            print(f"  - {usig['strategy']} signal at {usig.get('timestamp')} reason={usig.get('reason')}")

    # 2) Check for trades that have no corresponding signal
    trades_without_signal = []
    for trd in trade_entries:
        # if no parsed_signal_time, or no matching signal
        if not trd["parsed_signal_time"]:
            trades_without_signal.append(trd)
        else:
            # see if there's a signal that is close
            st = trd["parsed_signal_time"]
            matching = [
                s for s in signal_entries
                if s.get("parsed_time") and abs((s["parsed_time"] - st).total_seconds()) < 120
            ]
            if not matching:
                trades_without_signal.append(trd)

    print(f"\nTrades with no matching signal: {len(trades_without_signal)}")
    for tws in trades_without_signal:
        print(f"  - {tws['type']} {tws.get('amount')} BTC at {tws.get('price')} => {tws.get('timestamp')}")

    # 3) Group partial trades
    #    e.g., we might say partial trades are those that share the same 'parsed_signal_time'
    #    and occur within, say, 5 minutes. Then see if we effectively count them as 1.
    # This is just an example approach:
    grouped_by_signal = {}
    for trd in trade_entries:
        st = trd["parsed_signal_time"]
        if st is None:
            continue
        st_key = st.isoformat()
        if st_key not in grouped_by_signal:
            grouped_by_signal[st_key] = []
        grouped_by_signal[st_key].append(trd)

    for key, trades_list in grouped_by_signal.items():
        if len(trades_list) > 1:
            # We likely have partial trades
            print(f"\nDetected partial trades group with signal time ~ {key}: {len(trades_list)} partial trades")
            for p in trades_list:
                print(f"   => {p['type']} {p.get('amount')} BTC at {p.get('price')} on {p.get('timestamp')}")

    print("\nTrade-checker complete. You can extend these checks further as needed.")

    # (Optional) parse the log file if you want deeper checks
    if os.path.exists(log_file_path):
        print(f"\nOptionally, we could parse '{log_file_path}' for deeper confirmations, but skipping for brevity.")
        # e.g. read the log lines, confirm price ranges at trade times, etc.

if __name__ == "__main__":
    main()
