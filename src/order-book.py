import hashlib
import hmac
import time
import requests
import uuid
import json
import argparse
import logging

# Command line arguments
parser = argparse.ArgumentParser(description='Process Bitstamp orders.')
parser.add_argument('currency_pair', type=str, help='Currency pair')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
args = parser.parse_args()

# Read API configuration from file
with open('.bitstamp', 'r') as f:
    config = json.load(f)
api_key = config['api_key']
API_SECRET = bytes(config['api_secret'], 'utf-8')

timestamp = str(int(round(time.time() * 1000)))
nonce = str(uuid.uuid4())

message = 'BITSTAMP ' + api_key + \
    'GET' + \
    'www.bitstamp.net' + \
    '/api/v2/order_book/' + args.currency_pair + '/'
message = message.encode('utf-8')
signature = hmac.new(API_SECRET, msg=message, digestmod=hashlib.sha256).hexdigest()
headers = {
    'X-Auth': 'BITSTAMP ' + api_key,
    'X-Auth-Signature': signature,
    'X-Auth-Nonce': nonce,
    'X-Auth-Timestamp': timestamp,
    'X-Auth-Version': 'v2'
}

# Set up logging
logging.basicConfig(filename='bitstamp.log', level=logging.INFO)

url = 'https://www.bitstamp.net/api/v2/order_book/' + args.currency_pair + '/'
if args.verbose:
    print(f"Request Method: GET")
    print(f"Request URL: {url}")
    print(f"Request Headers: {headers}")
else:
    logging.info(f"Request Method: GET")
    logging.info(f"Request URL: {url}")
    logging.info(f"Request Headers: {headers}")

r = requests.get(
    url,
    headers=headers,
)

if not r.status_code == 200:
    logging.error(f"Unexpected status code: {r.status_code}")
    logging.error(f"Response text: {r.text}")
    raise Exception('Status code not 200')

if args.verbose:
    print(f"Response: {r.content}")
else:
    logging.info(f"Response: {r.content}")
print(r.content)
