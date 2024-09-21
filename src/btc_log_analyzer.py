import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def parse_log_file(file_path):
    data = []
    total_lines = sum(1 for _ in open(file_path, 'r'))
    print(f"Total lines in log file: {total_lines}")
    
    with open(file_path, 'r') as file:
        for i, line in enumerate(file, 1):
            if i % 10000 == 0:  # Update progress every 10,000 lines
                print(f"Processing line {i}/{total_lines} ({i/total_lines*100:.2f}%)")
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
    
    print("Finished processing log file. Creating DataFrame...")
    return pd.DataFrame(data)

def analyze_data(df):
    print("Converting timestamp to datetime...")
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    print("Calculating basic statistics...")
    print(df.describe())
    
    print("Plotting price over time...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['price'])
    plt.title('Bitcoin Price Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.savefig('btc_price_over_time.png')
    plt.close()
    
    print("Calculating and plotting hourly trading volume...")
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
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
    print(f"Log file size: {file_size:.2f} MB")
    
    print("Starting to parse log file...")
    df = parse_log_file(file_path)
    print(f"Parsed {len(df)} trade events.")
    
    print("Starting data analysis...")
    analyze_data(df)
    print("Analysis complete. Check the current directory for generated PNG files.")

if __name__ == "__main__":
    main()