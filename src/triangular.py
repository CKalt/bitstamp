#!env/bin/python
# src/triangle.py
import os
import pandas as pd
import logging

# Setting up logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='logs/triangular.log')

# Trade file setup
last_trade_timestamp = None
executed_trades = []

trade_file = 'trades/trades.txt'
if not os.path.exists('trades'):
    os.makedirs('trades')

def detect_opportunity(all_data, prices, last_updates, stale_counts, transaction_fee, ttl, profit_threshold, verbose):
    profitable_events = []
    for _, row in all_data.iterrows():
        event_pair = row['pair']
        event_price = row['price']
        event_timestamp = row['timestamp']

        if verbose:
            print(f"Processing event at {event_timestamp} for {event_pair} with price {event_price:.8f}")

        prices[event_pair] = event_price
        last_updates[event_pair] = event_timestamp

        is_bchbtc_stale = (event_timestamp - last_updates['bchbtc']).seconds > ttl if last_updates['bchbtc'] else True
        is_bchusd_stale = (event_timestamp - last_updates['bchusd']).seconds > ttl if last_updates['bchusd'] else True
        is_btcusd_stale = (event_timestamp - last_updates['btcusd']).seconds > ttl if last_updates['btcusd'] else True

        # Increment the stale counts based on staleness checks
        if is_bchbtc_stale:
            stale_counts['bchbtc'] += 1
        if is_bchusd_stale:
            stale_counts['bchusd'] += 1
        if is_btcusd_stale:
            stale_counts['btcusd'] += 1

        if is_bchbtc_stale or is_bchusd_stale or is_btcusd_stale:
            if verbose:
                print(f"Skipping due to stale price data. BCHBTC: {'Stale' if is_bchbtc_stale else 'Fresh'}, BCHUSD: {'Stale' if is_bchusd_stale else 'Fresh'}, BTCUSD: {'Stale' if is_btcusd_stale else 'Fresh'}")
            continue

        # Arbitrage calculation
        bch_after_conversion = (1.0 / prices['bchbtc']) * (1 - transaction_fee)
        usd_after_conversion = bch_after_conversion * prices['bchusd'] * (1 - transaction_fee)
        final_btc = usd_after_conversion / prices['btcusd'] * (1 - transaction_fee)
        profit_or_loss = final_btc - 1.0
        
        if profit_or_loss > profit_threshold:
            profitable_events.append({
                'timestamp': event_timestamp,
                'profit_or_loss': profit_or_loss,
                'bchbtc_price': prices['bchbtc'],
                'bchusd_price': prices['bchusd'],
                'btcusd_price': prices['btcusd']
            })
            
    return profitable_events



def execute_trade(profitable_events, trade_file, transaction_fee, last_updates, verbose):
    global last_trade_timestamp
    SKIP_TRADE_DELAY = 60  # 60 seconds
    
    trade_number = 1
    for event in profitable_events:
        if last_trade_timestamp and (event['timestamp'].timestamp() - last_trade_timestamp) < SKIP_TRADE_DELAY:
            continue
        last_trade_timestamp = event['timestamp'].timestamp()
        # Append to trades file
        with open(trade_file, 'a') as f:
            f.write(f"Trade {trade_number}: Timestamp (Epoch): {event['timestamp'].timestamp()}\n")
            f.write(f"Trade {trade_number}: Timestamp (Human): {event['timestamp']}\n")

            f.write("-" * 50 + "\n")
            f.write(f"1. Trading 1 BTC for BCH using {event['bchbtc_price']:.8f} BCH/BTC\n")
            f.write(f"   Timestamp (Epoch) for BCH/BTC price: {last_updates['bchbtc'].timestamp()}\n")
            bch_amount = (1.0 / event['bchbtc_price']) * (1 - transaction_fee)
            f.write(f"   After Trade (minus fee): {bch_amount:.8f} BCH\n")
            
            f.write("-" * 50 + "\n")
            f.write(f"2. Trading {bch_amount:.8f} BCH for USD using {event['bchusd_price']:.2f} BCH/USD\n")
            f.write(f"   Timestamp (Epoch) for BCH/USD price: {last_updates['bchusd'].timestamp()}\n")
            usd_amount = bch_amount * event['bchusd_price'] * (1 - transaction_fee)
            f.write(f"   After Trade (minus fee): ${usd_amount:.2f}\n")
            
            f.write("-" * 50 + "\n")
            f.write(f"3. Trading ${usd_amount:.2f} for BTC using {event['btcusd_price']:.2f} USD/BTC\n")
            f.write(f"   Timestamp (Epoch) for BTC/USD price: {last_updates['btcusd'].timestamp()}\n")
            final_btc_amount = usd_amount / event['btcusd_price'] * (1 - transaction_fee)
            f.write(f"   After Trade (minus fee): {final_btc_amount:.8f} BTC\n")
            
            f.write("-" * 50 + "\n")
            f.write(f"Profit for Trade {trade_number}: {event['profit_or_loss']:.8f} BTC\n")
            f.write("=" * 50 + "\n\n")

            executed_trades.append(event)

        logging.info(f"Profitable event detected at {event['timestamp']} with profit {event['profit_or_loss']:.8f} BTC")
        trade_number += 1

def refactored_triangular_arbitrage(bchbtc_path, bchusd_path, btcusd_path, profit_threshold=0.0, transaction_fee=0.002, ttl=60, verbose=True):
    # Load the datasets and convert timestamp column
    bchbtc_data = pd.read_csv(bchbtc_path).assign(
        timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='s'))
    bchusd_data = pd.read_csv(bchusd_path).assign(
        timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='s'))
    btcusd_data = pd.read_csv(btcusd_path).assign(
        timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='s'))

    all_data = pd.concat([
        bchbtc_data.assign(pair='bchbtc'),
        bchusd_data.assign(pair='bchusd'),
        btcusd_data.assign(pair='btcusd')
    ]).sort_values(by='timestamp')

    last_updates = {
        'bchbtc': None,
        'bchusd': None,
        'btcusd': None
    }

    prices = {
        'bchbtc': None,
        'bchusd': None,
        'btcusd': None
    }

    # Initialize the counts for stale data
    stale_counts = {
        'bchbtc': 0,
        'bchusd': 0,
        'btcusd': 0
    }

    profitable_events = detect_opportunity(all_data, prices, last_updates, stale_counts, transaction_fee, ttl, profit_threshold, verbose)
    execute_trade(profitable_events, trade_file, transaction_fee, last_updates, verbose)

    if verbose:
        for event in profitable_events:
            print(f"Timestamp: {event['timestamp']}")
            print(f"BCH/BTC Price: {event['bchbtc_price']:.8f} | BCH Obtained: {(1.0 / event['bchbtc_price']) * (1 - transaction_fee):.8f}")
            print(f"BCH/USD Price: {event['bchusd_price']:.2f} | USD Obtained: {((1.0 / event['bchbtc_price']) * (1 - transaction_fee)) * event['bchusd_price'] * (1 - transaction_fee):.2f}")
            print(f"BTC/USD Price: {event['btcusd_price']:.2f} | Final BTC: {(((1.0 / event['bchbtc_price']) * (1 - transaction_fee)) * event['bchusd_price'] * (1 - transaction_fee)) / event['btcusd_price'] * (1 - transaction_fee):.8f}")
            print(f"Profit: {event['profit_or_loss']:.8f} BTC")
            print("-" * 50)

    # Reporting the stale counts before the summary
    for pair, count in stale_counts.items():
        logging.info(f"{pair} was stale {count} times.")

    # Summary
    with open(trade_file, 'a') as f:
        total_opportunities = len(executed_trades)
        total_profit = sum([event['profit_or_loss'] for event in executed_trades])
        f.write(f"Total Opportunities: {total_opportunities}\n")
        f.write(f"Total Profit: {total_profit:.8f} BTC\n")

    logging.info(f"Total Opportunities: {total_opportunities}")
    logging.info(f"Total Profit: {total_profit:.8f} BTC")

if __name__ == "__main__":
    # Clearing existing trade file for a new run
    if os.path.exists(trade_file):
        os.remove(trade_file)

    bchbtc_path = 'bchbtc.csv'
    bchusd_path = 'bchusd.csv'
    btcusd_path = 'btcusd.csv'

    refactored_triangular_arbitrage(bchbtc_path, bchusd_path, btcusd_path,
                         profit_threshold=0.0001, transaction_fee=0.002, ttl=60, verbose=True)
