from datetime import datetime
from influxdb_client import WritePrecision, InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json

f = open('.influxdb', 'r')
config = json.load(f)

url = config['url']
token = config['token']
org = config['org']
bucket = config['bucket']

print(f"bucket = {bucket}")

client =  InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

query = f'from(bucket: "{bucket}") |> range(start: -30y) |> sort(columns: ["_time"], desc: true)'

tables = query_api.query(query)

records = []
for table in tables:
    for record in table.records:
        records.append(record)

if records:
    print(f"Latest time: {records[0].get_time()}")
    print(f"Oldest time: {records[-1].get_time()}")
else:
    print("No data found in the specified range.")
