import requests
import pandas as pd
from datetime import datetime, timedelta
from influxdb_client import WritePrecision, InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

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

# Query data back from InfluxDB to get the most recent timestamp
query_api = client.query_api()
tables = query_api.query(f'from(bucket: "{bucket}") |> range(start: -1d) |> last()')

for table in tables:
    for record in table.records:
        last_timestamp = record["_time"].replace(tzinfo=None)  # make the timestamp "naive"

# Get the current time (in UTC for consistency) and round down to nearest minute
current_time = datetime.utcnow()
current_time = current_time.replace(second=0, microsecond=0)

# Calculate difference in minutes
minutes_diff = int((current_time - last_timestamp).total_seconds() / 60)

# If the difference is more than 1 minute
if minutes_diff >= 1:
    params_bitstamp = {
        'step': 60,  # Step size in seconds.
        'limit': minutes_diff  # Number of data points.
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

        print(f"Added {len(df)} new bars to the database.")
    else:
        print(f"Failed to get data: {response.content}")
else:
    next_minute = 60 - datetime.utcnow().second
    print(f"The last minute bar stored is sufficient and the database need not be updated. Check again in {next_minute} seconds.")
