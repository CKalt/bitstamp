import pandas as pd

def triangular_arbitrage(bchbtc_path, bchusd_path, btcusd_path, profit_threshold=0.0, transaction_fee=0.002, bar_minutes=1):
    # Load the datasets
    bchbtc_data = pd.read_csv(bchbtc_path, parse_dates=['timestamp'])
    bchusd_data = pd.read_csv(bchusd_path, parse_dates=['timestamp'])
    btcusd_data = pd.read_csv(btcusd_path, parse_dates=['timestamp'])

    # Resample ticker data to create bars
    bchbtc_data = bchbtc_data.resample(f'{bar_minutes}T', on='timestamp').agg({'price': ['mean', 'max', 'min']}).reset_index()
    bchusd_data = bchusd_data.resample(f'{bar_minutes}T', on='timestamp').agg({'price': ['mean', 'max', 'min']}).reset_index()
    btcusd_data = btcusd_data.resample(f'{bar_minutes}T', on='timestamp').agg({'price': ['mean', 'max', 'min']}).reset_index()

    # Flatten multi-level column names
    bchbtc_data.columns = ['timestamp', 'price_bchbtc', 'price_high_bchbtc', 'price_low_bchbtc']
    bchusd_data.columns = ['timestamp', 'price_bchusd', 'price_high_bchusd', 'price_low_bchusd']
    btcusd_data.columns = ['timestamp', 'price_btcusd', 'price_high_btcusd', 'price_low_btcusd']

    # Merge datasets on timestamp
    merged_data = bchbtc_data.merge(bchusd_data, on='timestamp', how='inner')
    merged_data = merged_data.merge(btcusd_data, on='timestamp', how='inner')

    # Arbitrage calculation
    bch_after_conversion = (1.0 / merged_data['price_high_bchbtc']) * (1 - transaction_fee)
    usd_after_conversion = bch_after_conversion * merged_data['price_low_bchusd'] * (1 - transaction_fee)
    final_btc = usd_after_conversion / merged_data['price_high_btcusd'] * (1 - transaction_fee)
    merged_data['profit_or_loss'] = final_btc - 1.0

    # Filter for profitable trades
    opportunities = merged_data[merged_data['profit_or_loss'] > profit_threshold]
    for _, row in opportunities.iterrows():
        print(f"Timestamp: {row['timestamp']}")
        print(f"BCH/BTC High Price: {row['price_high_bchbtc']:.8f} | BCH Obtained: {bch_after_conversion.iloc[0]:.8f}")
        print(f"BCH/USD Low Price: {row['price_low_bchusd']:.2f} | USD Obtained: {usd_after_conversion.iloc[0]:.2f}")
        print(f"BTC/USD High Price: {row['price_high_btcusd']:.2f} | Final BTC: {final_btc.iloc[0]:.8f}")
        print(f"Profit: {row['profit_or_loss']:.8f} BTC")
        print("-" * 50)

    # Summary
    total_opportunities = len(opportunities)
    total_profit = opportunities['profit_or_loss'].sum()
    print(f"Total Opportunities: {total_opportunities}")
    print(f"Total Profit: {total_profit:.8f} BTC")

if __name__ == "__main__":
    bchbtc_path = 'bchbtc_small.csv'
    bchusd_path = 'bchusd_small.csv'
    btcusd_path = 'btcusd_small.csv'

    triangular_arbitrage(bchbtc_path, bchusd_path, btcusd_path, profit_threshold=0.0001, transaction_fee=0.002, bar_minutes=1)
