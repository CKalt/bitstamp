#! env/bin/python
# src/buy-order.py
import hashlib
import hmac
import time
import requests
import uuid
import sys
import json
import argparse
import logging

# Command line arguments
parser = argparse.ArgumentParser(description='Process Bitstamp orders.')
parser.add_argument('order_file', type=str, help='Order file')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
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

order_type_endpoint = {
    "instant-buy": "buy/instant",
    "market-buy": "buy/market",
    "limit-buy": "buy/limit"
}
url_endpoint = order_type_endpoint.get(order['order_type'])
if not url_endpoint:
    raise ValueError("Unsupported order type")

if order['order_type'] == "limit-buy":
    payload = {
        'amount': str(order['amount']),
        'price': str(order['price'])
        # ... add other fields like 'fok_order', 'gtd_order', 'ioc_order', 'limit_price', 'moc_order' based on the order JSON if they exist
    }
elif order['order_type'] in ["instant-buy", "market-buy"]:
    payload = {
        'amount': str(order['amount'])
    }
    if 'client_order_id' in order:
        payload['client_order_id'] = order['client_order_id']
else:
    raise ValueError("Unsupported order type for payload")

from urllib.parse import urlencode
payload_string = urlencode(payload)

message = 'BITSTAMP ' + api_key + \
    'POST' + \
    'www.bitstamp.net' + \
    f'/api/v2/{url_endpoint}/{order["currency_pair"]}/' + \
    '' + \
    content_type + \
    nonce + \
    timestamp + \
    'v2' + \
    payload_string
message = message.encode('utf-8')
signature = hmac.new(API_SECRET, msg=message, digestmod=hashlib.sha256).hexdigest()
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

url = f'https://www.bitstamp.net/api/v2/{url_endpoint}/{order["currency_pair"]}/'
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
    logging.error(f"Unexpected status code: {r.status_code}")
    logging.error(f"Response text: {r.text}")
    raise Exception('Status code not 200')

string_to_sign = (nonce + timestamp + r.headers.get('Content-Type')).encode('utf-8') + r.content
signature_check = hmac.new(API_SECRET, msg=string_to_sign, digestmod=hashlib.sha256).hexdigest()
if not r.headers.get('X-Server-Auth-Signature') == signature_check:
    logging.error('Signatures do not match')
    raise Exception('Signatures do not match')

if args.verbose:
    print(f"Response: {r.content}")
else:
    logging.info(f"Response: {r.content}")
print(r.content)
