###############################################################################
# src/trade-checker.py
###############################################################################
# Full File Path: src/trade-checker.py
#
# CHANGES (EXPLANATION):
#   1) This is a new file, so there were no original docstrings to preserve, 
#      but we keep minimal logic plus some clarifying comments.
#   2) We group partial trades by the same signal_timestamp or a short time range.
#   3) We match signals to trades, detect unmatched signals or trades without signals.
#   4) Print a summary so you can see if your auto_trade is performing as expected.
###############################################################################

import json
import os
from datetime import datetime, timedelta

def load_signals(signal_file="signal_logs.json"):
    """
    Load signals from signal_logs.json (or another file).
    Return a list of dict with keys like:
       timestamp, signal_value, strategy, reason, ...
    """
    if not os.path.exists(signal_file):
        print(f"No signal file '{signal_file}'.")
        return []
    with open(signal_file, 'r') as f:
        try:
            signals = json.load(f)
        except json.JSONDecodeError:
            signals = []
    return signals


def load_trades(trade_file="non-live-trades.json"):
    """
    Load trades from a trades JSON file (non-live or live).
    Return a list of dict with keys like:
       type, symbol, amount, price, timestamp, ...
    """
    if not os.path.exists(trade_file):
        print(f"No trade file '{trade_file}'.")
        return []
    with open(trade_file, 'r') as f:
        try:
            trades = json.load(f)
        except json.JSONDecodeError:
            trades = []
    return trades


def parse_iso_or_timestr(ts_str):
    """
    Attempt to parse a timestamp in various formats to a datetime object.
    """
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            pass
    # fallback: 
    return None


def group_partial_trades(trades, time_diff_seconds=5):
    """
    Group partial trades that share the same 'signal_timestamp' or occur
    within a short time_diff_seconds of each other with identical 'reason'.
    Return a structure like:
        [
          {
            "signal_time": <datetime or None>,
            "reason": "RSI < 30.0" (for instance),
            "trades": [list of sub-trade dicts]
          },
          ...
        ]
    """
    grouped = []
    used = set()

    # Convert trade timestamps to datetime
    for t in trades:
        t['parsed_timestamp'] = parse_iso_or_timestr(t.get('signal_timestamp') or t.get('timestamp'))
        # The 'signal_timestamp' is typically how we link them to the signals,
        # but if missing, we fallback to the actual trade 'timestamp'.

    trades_sorted = sorted(trades, key=lambda x: x['parsed_timestamp'] if x['parsed_timestamp'] else datetime.min)

    for i, trade in enumerate(trades_sorted):
        if i in used:
            continue
        group = {
            "signal_time": trade.get('signal_timestamp'),
            "reason": trade.get('reason'),
            "trades": [trade]
        }
        used.add(i)

        # Check subsequent trades that might share the same reason & close timestamp
        for j in range(i+1, len(trades_sorted)):
            if j in used:
                continue
            t2 = trades_sorted[j]
            # same reason?
            if t2.get('reason') != trade.get('reason'):
                continue
            # same or close signal_timestamp?
            st1 = parse_iso_or_timestr(trade.get('signal_timestamp') or trade.get('timestamp'))
            st2 = parse_iso_or_timestr(t2.get('signal_timestamp') or t2.get('timestamp'))
            if st1 and st2:
                diff = abs((st2 - st1).total_seconds())
                if diff <= time_diff_seconds:
                    group["trades"].append(t2)
                    used.add(j)

        grouped.append(group)
    return grouped


def match_signals_to_trades(signals, grouped_trades):
    """
    Attempt to pair each signal to one of the grouped trades by matching
    signal timestamp to group["signal_time"] or reason/time proximity.
    Return lists of matched signals, unmatched signals, and trades lacking signals.
    """
    matched_signals = []
    unmatched_signals = []
    trades_with_signal = set()

    # Convert signals to a workable list
    for s in signals:
        s['parsed_timestamp'] = parse_iso_or_timestr(s.get('timestamp'))

    # We'll do a simple approach: if the group's "signal_time" is close to the signal's timestamp
    # or exactly matches, we consider them matched.
    for sig in signals:
        st = sig.get('parsed_timestamp')
        if not st:
            continue
        matched = False
        for i, grp in enumerate(grouped_trades):
            gst = parse_iso_or_timestr(grp["signal_time"])
            if not gst:
                continue
            diff = abs((st - gst).total_seconds())
            if diff <= 5:  # or exact match
                matched_signals.append(sig)
                trades_with_signal.add(i)
                matched = True
                break
        if not matched:
            unmatched_signals.append(sig)

    trades_without_signal = []
    for i, grp in enumerate(grouped_trades):
        if i not in trades_with_signal:
            # no signal matched this trade group
            trades_without_signal.append(grp)

    return matched_signals, unmatched_signals, trades_without_signal


def main():
    signal_file = "signal_logs.json"
    trade_file = "non-live-trades.json"  # or "trades.json" if you're live

    signals = load_signals(signal_file)
    trades = load_trades(trade_file)

    print(f"Loaded {len(signals)} signals from {os.path.abspath(signal_file)}")
    print(f"Loaded {len(trades)} trades from {os.path.abspath(trade_file)}\n")

    grouped = group_partial_trades(trades, time_diff_seconds=10)

    matched_signals, unmatched_signals, trades_no_signal = match_signals_to_trades(signals, grouped)

    print(f"Signals matched to at least one trade: {len(matched_signals)}")
    if unmatched_signals:
        print(f"Unmatched signals (no trades found): {len(unmatched_signals)}")
        for usig in unmatched_signals:
            uts = usig.get('timestamp')
            reason = usig.get('reason')
            print(f"  - {usig.get('strategy','FORCED')} signal at {uts} reason={reason}")
    else:
        print("Unmatched signals: 0")

    if trades_no_signal:
        print(f"\nTrades with no matching signal: {len(trades_no_signal)}")
        for grp in trades_no_signal:
            if not grp["trades"]:
                continue
            # Print a summary
            first_trade = grp["trades"][0]
            print(f"  - {first_trade['type']} {first_trade['amount']} BTC at {first_trade['price']} => {first_trade['timestamp']}")
    else:
        print("\nNo trades lacked signals.")

    print("\nDetected partial trades group(s):")
    for grp in grouped:
        st = grp.get("signal_time")
        reason = grp.get("reason")
        subtrades = grp.get("trades", [])
        if len(subtrades) > 1:
            print(f"  => Found partial trades group with ~ signal time {st}: {len(subtrades)} partial trades")
            for t in subtrades:
                print(f"     => {t['type']} {t['amount']} BTC at {t['price']} on {t['timestamp']}")
        else:
            # single trade group
            t = subtrades[0]
            print(f"  => Single trade group for {t['type']} {t['amount']} BTC at {t['price']} on {t['timestamp']}")

    print("\nTrade-checker complete. You can extend these checks further as needed.")


if __name__ == "__main__":
    main()
