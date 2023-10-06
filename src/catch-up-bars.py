import requests
import pandas as pd
import argparse
from datetime import datetime, timedelta
import psycopg2
import os
import traceback

try:
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
    print("Parsing arguments...")
    args = parser.parse_args()
    print(f"Arguments: {args}")

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
    print("Querying PostgreSQL for timestamps...")
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
    print(f"Last timestamp: {last_timestamp}, Oldest timestamp: {oldest_timestamp}")

    # Check if last_timestamp or oldest_timestamp is None
    if last_timestamp is not None:
        last_timestamp = last_timestamp[0]
    if oldest_timestamp is not None:
        oldest_timestamp = oldest_timestamp[0]

    # Calculate the time difference in seconds
    seconds_diff = None  # Initialize seconds_diff to None
    if last_timestamp and oldest_timestamp:
        seconds_diff = int((datetime.utcnow() - last_timestamp).total_seconds())

    start_time = last_timestamp if last_timestamp else datetime.utcnow() - timedelta(minutes=1)
    end_time = datetime.utcnow()

    limit_value = int((end_time - start_time).total_seconds() / 60)
    if limit_value > 1000:
        limit_value = 1000

    # Fetch and store data since last bar if no options are provided
    elif not args.o and not args.d and args.b == 0:
        print("Fetching and storing data since last bar...")

        params_bitstamp = {
            'start': str(int(start_time.timestamp())),
            'end': str(int(end_time.timestamp())),
            'step': 60,  # Step size in seconds.
            'limit': limit_value  # Adjusted limit.
        }

        # Make a GET request to the Bitstamp API
        response = requests.get(url_bitstamp, params=params_bitstamp)

        if response.status_code == 200:
            # Load response JSON
            data = response.json()

            if not data['data']['ohlc']:
                print("No new data available to insert.")
            else:
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
                print(f"Added {len(df)} new bars to the database since {start_time}.")
        else:
            print(f"API request failed with status code {response.status_code}.")

    # If only the -o option is specified, show the date and time of the oldest bar stored in the PostgreSQL database and how many days ago that was.
    elif args.o and not args.d and args.b == 0:
        print("Showing the oldest bar...")
        if oldest_timestamp:
            days_ago = (datetime.utcnow() - oldest_timestamp).days
            print(f"The oldest bar is from {oldest_timestamp} which is {days_ago} days ago.")
        else:
            print("No data in the database yet.")

    print("Execution completed.")

except Exception as e:
    print("Exception occurred: ", str(e))
    traceback.print_exc()
