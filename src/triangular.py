import pandas as pd

def triangular_arbitrage(bchbtc_path, bchusd_path, btcusd_path, profit_threshold=0.0, transaction_fee=0.002, ttl=60, verbose=True):
    # Load the datasets and convert timestamp column
    bchbtc_data = pd.read_csv(bchbtc_path).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='s'))
    bchusd_data = pd.read_csv(bchusd_path).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='s'))
    btcusd_data = pd.read_csv(btcusd_path).assign(timestamp=lambda x: pd.to_datetime(x['timestamp'], unit='s'))
    
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
            if verbose:
                print(f"Profitable event detected at {event_timestamp} with profit {profit_or_loss:.8f} BTC")
        else:
            if verbose:
                print(f"No arbitrage opportunity at {event_timestamp}")

    if verbose:
        for event in profitable_events:
            print(f"Timestamp: {event['timestamp']}")
            print(f"BCH/BTC Price: {event['bchbtc_price']:.8f} | BCH Obtained: {(1.0 / event['bchbtc_price']) * (1 - transaction_fee):.8f}")
            print(f"BCH/USD Price: {event['bchusd_price']:.2f} | USD Obtained: {((1.0 / event['bchbtc_price']) * (1 - transaction_fee)) * event['bchusd_price'] * (1 - transaction_fee):.2f}")
            print(f"BTC/USD Price: {event['btcusd_price']:.2f} | Final BTC: {(((1.0 / event['bchbtc_price']) * (1 - transaction_fee)) * event['bchusd_price'] * (1 - transaction_fee)) / event['btcusd_price'] * (1 - transaction_fee):.8f}")
            print(f"Profit: {event['profit_or_loss']:.8f} BTC")
            print("-" * 50)

    # Summary
    total_opportunities = len(profitable_events)
    total_profit = sum([event['profit_or_loss'] for event in profitable_events])
    print(f"Total Opportunities: {total_opportunities}")
    print(f"Total Profit: {total_profit:.8f} BTC")

if __name__ == "__main__":
    bchbtc_path = 'bchbtc_small.csv'
    bchusd_path = 'bchusd_small.csv'
    btcusd_path = 'btcusd_small.csv'

    triangular_arbitrage(bchbtc_path, bchusd_path, btcusd_path, profit_threshold=0.0001, transaction_fee=0.002, ttl=60, verbose=True)
