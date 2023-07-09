import requests
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

url = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'
params = {
    'step': '60',
    'limit': '3'
}
response = requests.get(url, params=params)
data = response.json()
df = pd.DataFrame(data['data']['ohlc'])

df[['close', 'high', 'low', 'open', 'volume']] = df[['close', 'high', 'low', 'open', 'volume']].apply(pd.to_numeric)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# You can generate a Token from the "Tokens Tab" in the UI
token = "<my-token>"
org = "<my-org>"
bucket = "<my-bucket>"

client = InfluxDBClient(url="http://localhost:8086", token=token)

write_api = client.write_api(write_options=SYNCHRONOUS)

for index, row in df.iterrows():
    point = Point("ohlc")\
        .tag("host", "host1")\
        .field("close", row['close'])\
        .field("high", row['high'])\
        .field("low", row['low'])\
        .field("open", row['open'])\
        .field("volume", row['volume'])\
        .time(row['timestamp'], WritePrecision.NS)

    write_api.write(bucket, org, point)

write_api.__del__()
client.__del__()
