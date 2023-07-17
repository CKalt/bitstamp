import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
import psycopg2
import os

# Read the PostgreSQL environment variables
database = os.environ.get('PGDATABASE')
host = os.environ.get('PGHOST')
port = os.environ.get('PGPORT')
user = os.environ.get('PGUSER')

# Create a connection to the PostgreSQL database
conn = psycopg2.connect(
    database=database,
    host=host,
    port=port,
    user=user
)

# Create a cursor object to execute SQL statements
cur = conn.cursor()

# Bitstamp parameters
url_bitstamp = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Fetch and store Bitcoin price data.')
parser.add_argument('-o', action='store_true',
                    help='Show the date and time of the oldest bar stored in the PostgreSQL database and how many days ago that was.')
parser.add_argument('-b', type=int, default=0,
                    help='Number of additional days to request from Bitstamp and then store.')
parser.add_argument('-d', action='store_true',
                    help='Perform a dry run. If provided with the -b option, it will report on the operations that would take place but will not actually perform them.')
parser.add_argument('--initdb', action='store_true',
                    help='Perform the necessary DDL statements and create the required tables.')
args = parser.parse_args()

# Perform DDL statements if --initdb option is specified
if args.initdb:
    # Create the required table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bitstamp (
            id SERIAL PRIMARY KEY,
            currency VARCHAR(10) NOT NULL,
            open FLOAT NOT NULL,
            close FLOAT NOT NULL,
            low FLOAT NOT NULL,
            high FLOAT NOT NULL,
            volume FLOAT NOT NULL,
            timestamp TIMESTAMPTZ NOT NULL
        )
    """)
    conn.commit()
    print("DDL statements executed successfully.")

# Query data back from PostgreSQL to get the most recent and oldest timestamp
cur.execute("""
    SELECT timestamp FROM bitstamp
    ORDER BY timestamp DESC
    LIMIT 1
""")
last_timestamp = cur.fetchone()

cur.execute("""
    SELECT timestamp FROM bitstamp
    ORDER BY timestamp ASC
    LIMIT 1
""")
oldest_timestamp = cur.fetchone()

# Check if last_timestamp or oldest_timestamp is None
if last_timestamp is not None:
    last_timestamp = last_timestamp[0]
if oldest_timestamp is not None:
    oldest_timestamp = oldest_timestamp[0]

# Calculate the time difference in seconds
seconds_diff = None  # Initialize seconds_diff to None
if last_timestamp and oldest_timestamp:
    seconds_diff = int((datetime.utcnow() - last_timestamp).total_seconds())

# If only the -d option is specified, report the range of data that would be fetched
if args.d and not args.o and args.b == 0:
    # Calculate the start and end times for the request
    start_time = last_timestamp
    end_time = datetime.utcnow()

    limit_value = int((end_time - start_time).total_seconds() / 60)
    if limit_value > 1000:
        limit_value = 1000

    print(f"(Dry Run) Would fetch data from {start_time} to {end_time} with limit: {limit_value}")

# If only the -o option is specified, show the date and time of the oldest bar stored in the PostgreSQL database and how many days ago that was.
elif args.o and not args.d and args.b == 0:
    if oldest_timestamp:
        days_ago = (datetime.utcnow() - oldest_timestamp).days
        print(f"The oldest bar is from {oldest_timestamp} which is {days_ago} days ago.")
    else:
        print("No data in the database yet.")

# If only the -b option is specified (greater than zero), fetch and store the requested number of additional days.
elif args.b > 0 and not args.o and not args.d:
    if oldest_timestamp:
        total_days = (datetime.utcnow() - oldest_timestamp).days
        extra_days = args.b
        # Only fetch data for the days not yet covered by the existing data
        for i in range(total_days + 1, total_days + extra_days + 1):
            # Calculate the start and end times for the request
            start_time = datetime.utcnow() - timedelta(days=i)

            # Check if the start time is older than the oldest timestamp in the database
            if oldest_timestamp and start_time < oldest_timestamp:
                print("Data for this period already exists in the database.")
                continue

            end_time = start_time + timedelta(days=1)

            params_bitstamp = {
                'start': str(int(start_time.timestamp())),
                'end': str(int(end_time.timestamp())),
                'step': 60,  # Step size in seconds.
                'limit': 1000  # Maximum limit.
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

                # Insert data into the PostgreSQL database
                for index, row in df.iterrows():
                    cur.execute("""
                        INSERT INTO bitstamp (currency, open, close, low, high, volume, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, ("btcusd", row['open'], row['close'], row['low'], row['high'], row['volume'], row['timestamp']))
                conn.commit()
                print(f"Added {len(df)} new bars to the database for {start_time.date()}.")


