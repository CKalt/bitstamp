import requests
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import WritePrecision, InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import time

# Load InfluxDB config
f = open('.influxdb', 'r')
config = json.load(f)

url = config['url']
token = config['token']
org = config['org']
bucket = config['bucket']

# Create InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Bitstamp parameters
url_bitstamp = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'
params_bitstamp = {
    'step': 60,  # Step size in seconds. 
    'limit': 5  # Number of data points.
}

# Make a GET request to the Bitstamp API
response = requests.get(url_bitstamp, params=params_bitstamp)

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

    # Write data to InfluxDB
    for index, row in df.iterrows():
        p = Point("bitstamp") \
            .tag("currency", "btcusd") \
            .field("open", row['open']) \
            .field("close", row['close']) \
            .field("low", row['low']) \
            .field("high", row['high']) \
            .field("volume", row['volume']) \
            .time(row['timestamp'], WritePrecision.MS)

        write_api.write(bucket=bucket, record=p)
else:
    print(f"Failed to get data: {response.content}")

# Query data back from InfluxDB
query_api = client.query_api()
tables = query_api.query(f'from(bucket: "{bucket}") |> range(start: -1d)')

for table in tables:
    for record in table.records:
        print(str(record["_time"]) + " - " + record.get_measurement()
                + " " + record.get_field() + "=" + str(record.get_value()))
