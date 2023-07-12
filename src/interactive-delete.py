import json
from influxdb_client import InfluxDBClient

def delete_measurement_records(measurement):
    # Read API configuration from file
    with open('.influxdb', 'r') as f:
        config = json.load(f)

    url = config['url']
    token = config['token']
    org = config['org']
    bucket = config['bucket']

    # Connect to InfluxDB
    with InfluxDBClient(url=url, token=token) as client:
        delete_api = client.delete_api()

        # Construct delete predicate for the given measurement
        predicate = f'_measurement="{measurement}"'

        # Set the start and stop time to delete the specific rows
        start = '2020-01-01T08:00:00Z'
        stop = '2029-01-01T20:00:00Z'

        # Delete records for the measurement within the specified time range
        delete_api.delete(start=start, stop=stop, predicate=predicate, bucket=bucket, org=org)

        print(f"All records for measurement '{measurement}' have been deleted.")

if __name__ == '__main__':
    measurement_to_delete = 'weatherstation'  # Replace with your desired measurement name
    delete_measurement_records(measurement_to_delete)
