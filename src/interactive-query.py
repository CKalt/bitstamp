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

tables = query_api.query(f'from(bucket: "{bucket}") |> range(start: -1d)')

for table in tables:
    for record in table.records:
        print(str(record["_time"]) + " - " + record.get_measurement()
                + " " + record.get_field() + "=" + str(record.get_value()))

