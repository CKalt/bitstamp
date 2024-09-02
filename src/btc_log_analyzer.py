import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_log_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                json_data = json.loads(line)
                if json_data['event'] == 'trade':
                    trade_data = json_data['data']
                    data.append({
                        'timestamp': int(trade_data['timestamp']),
                        'price': float(trade_data['price']),
                        'amount': float(trade_data['amount']),
                        'type': int(trade_data['type'])
                    })
            except json.JSONDecodeError:
                continue  # Skip lines that are not valid JSON
    return pd.DataFrame(data)

def analyze_data(df):
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Basic statistics
    print(df.describe())
    
    # Price over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['price'])
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.savefig('btc_price_over_time.png')
    plt.close()
    
    # Volume over time
    df['volume'] = df['price'] * df['amount']
    hourly_volume = df.resample('H', on='datetime')['volume'].sum()
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_volume.index, hourly_volume.values)
    plt.title('Hourly Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume (USD)')
    plt.savefig('btc_hourly_volume.png')
    plt.close()

def main():
    file_path = 'btcusd.log'
    df = parse_log_file(file_path)
    analyze_data(df)

if __name__ == "__main__":
    main()
