import requests
import pandas as pd

def get_ohlc_data():
    url = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'
    params = {
        'step': 60,  # Step size in seconds. 
        'limit': 3  # Limit is the number of data points.
    }
    
    # Make a GET request to the Bitstamp API
    response = requests.get(url, params=params)
    
    # Ensure the request was successful
    if response.status_code == 200:
        # Load response JSON
        data = response.json()
        
        # Create a DataFrame from the OHLC data
        df = pd.DataFrame(data['data']['ohlc'])
        
        # Convert timestamp from string to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s')
        
        # Convert other columns to appropriate data types
        df[['close', 'high', 'low', 'open']] = df[['close', 'high', 'low', 'open']].astype(float)
        df['volume'] = df['volume'].astype(float)

        return df
    else:
        print(f"Failed to get data: {response.content}")
        return None

df = get_ohlc_data()
print(df)
