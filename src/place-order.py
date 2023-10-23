#!env/bin/python
# src/place-orders.py
from urllib.parse import urlencode
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
parser.add_argument('order_file', type=str, help='Order file')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Verbose output')
args = parser.parse_args()

# Read API configuration from file
with open('.bitstamp', 'r') as f:
    config = json.load(f)
api_key = config['api_key']
API_SECRET = bytes(config['api_secret'], 'utf-8')

# Read order data from file
with open(args.order_file, 'r') as f:
    order = json.load(f)

timestamp = str(int(round(time.time() * 1000)))
nonce = str(uuid.uuid4())
content_type = 'application/x-www-form-urlencoded'

order_type = order.get('order_type', 'limit-buy')

if order_type == 'limit-buy':
    endpoint = '/api/v2/buy/'
    payload = {'amount': str(order['amount']), 'price': str(order['price'])}
elif order_type == 'limit-sell':
    endpoint = '/api/v2/sell/'
    payload = {'amount': str(order['amount']), 'price': str(order['price'])}
elif order_type == 'market-buy' or order_type == 'instant-buy':
    endpoint = '/api/v2/buy/market/'
    # Reading the price from the input JSON
    # Defaulting to '28113' if not provided
    price_value = str(order.get('price', '28113'))
    payload = {'amount': str(order['amount']), 'price': price_value}
    # Outputting the constructed JSON payload for verification
    print(f"Constructed Trade Payload: {json.dumps(payload)}")
elif order_type == 'market-sell' or order_type == 'instant-sell':
    endpoint = '/api/v2/sell/market/'
    # Reading the price from the input JSON
    # Defaulting to '28113' if not provided
    price_value = str(order.get('price', '28113'))
    payload = {'amount': str(order['amount']), 'price': price_value}
    # Outputting the constructed JSON payload for verification
    print(f"Constructed Trade Payload: {json.dumps(payload)}")
else:
    raise ValueError(f"Unsupported order type: {order_type}")

url = 'https://www.bitstamp.net' + endpoint + order['currency_pair'] + '/'


payload_string = urlencode(payload)

# '' (empty string) in message represents any query parameters or an empty string in case there are none
message = 'BITSTAMP ' + api_key + \
    'POST' + \
    'www.bitstamp.net' + \
    endpoint + order['currency_pair'] + '/' + \
    '' + \
    content_type + \
    nonce + \
    timestamp + \
    'v2' + \
    payload_string
message = message.encode('utf-8')
signature = hmac.new(API_SECRET, msg=message,
                     digestmod=hashlib.sha256).hexdigest()
headers = {
    'X-Auth': 'BITSTAMP ' + api_key,
    'X-Auth-Signature': signature,
    'X-Auth-Nonce': nonce,
    'X-Auth-Timestamp': timestamp,
    'X-Auth-Version': 'v2',
    'Content-Type': content_type
}

# Set up logging
logging.basicConfig(filename='bitstamp.log', level=logging.INFO)

url = 'https://www.bitstamp.net' + endpoint + order['currency_pair'] + '/'
if args.verbose:
    print(f"Request Method: POST")
    print(f"Request URL: {url}")
    print(f"Request Headers: {headers}")
    print(f"Request Payload: {payload_string}")
else:
    logging.info(f"Request Method: POST")
    logging.info(f"Request URL: {url}")
    logging.info(f"Request Headers: {headers}")
    logging.info(f"Request Payload: {payload_string}")

r = requests.post(
    url,
    headers=headers,
    data=payload_string
)

if not r.status_code == 200:
    print(f"Error Response from Bitstamp API: {r.text}")
    logging.error(f"Unexpected status code: {r.status_code}")
    logging.error(f"Response text: {r.text}")
    raise Exception('Status code not 200')

string_to_sign = (nonce + timestamp +
                  r.headers.get('Content-Type')).encode('utf-8') + r.content
signature_check = hmac.new(
    API_SECRET, msg=string_to_sign, digestmod=hashlib.sha256).hexdigest()
if not r.headers.get('X-Server-Auth-Signature') == signature_check:
    logging.error('Signatures do not match')
    raise Exception('Signatures do not match')

if args.verbose:
    print(f"Response: {r.content}")
else:
    logging.info(f"Response: {r.content}")
print(r.content)
