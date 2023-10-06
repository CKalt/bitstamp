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

client = InfluxDBClient(url=url, token=token, org=org)
p = Point("weatherstation") \
    .tag("location", "San Francisco") \
    .field("temperature", 25.9) \
    .time(datetime.utcnow(), WritePrecision.MS)

write_api = client.write_api(write_options=SYNCHRONOUS)
write_api.write(bucket=bucket, record=p)