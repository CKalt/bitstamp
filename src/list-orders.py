import hashlib
import hmac
import time
import requests
import uuid
import json
import argparse
import logging
from urllib.parse import urlencode

# Command line arguments
parser = argparse.ArgumentParser(description='List Open Bitstamp orders.')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
args = parser.parse_args()

# Read API configuration from file
with open('.bitstamp', 'r') as f:
    config = json.load(f)
api_key = config['api_key']
API_SECRET = bytes(config['api_secret'], 'utf-8')

# Setup request details
timestamp = str(int(round(time.time() * 1000)))
nonce = str(uuid.uuid4())

# Build the message for HMAC signature
message = 'BITSTAMP ' + api_key + \
    'POST' + \
    'www.bitstamp.net' + \
    '/api/v2/open_orders/all/' + \
    '' + \
    nonce + \
    timestamp + \
    'v2'

message = message.encode('utf-8')

# Generate the HMAC signature
signature = hmac.new(API_SECRET, msg=message, digestmod=hashlib.sha256).hexdigest()

headers = {
    'X-Auth': 'BITSTAMP ' + api_key,
    'X-Auth-Signature': signature,
    'X-Auth-Nonce': nonce,
    'X-Auth-Timestamp': timestamp,
    'X-Auth-Version': 'v2',
}

# Set up logging
logging.basicConfig(filename='bitstamp.log', level=logging.INFO if not args.verbose else logging.DEBUG)

logger = logging.getLogger()

url = 'https://www.bitstamp.net/api/v2/open_orders/all/'

logger.info(f"Request Method: POST")
logger.info(f"Request URL: {url}")
logger.info(f"Request Headers: {headers}")

r = requests.post(
    url,
    headers=headers,
)

logger.info(f"Response Status Code: {r.status_code}")
logger.info(f"Response Headers: {r.headers}")
logger.info(f"Response Body: {r.text}")

if not r.status_code == 200:
    logger.error(f"Unexpected status code: {r.status_code}")
    logger.error(f"Response text: {r.text}")
    raise Exception('Status code not 200')

string_to_sign = (nonce + timestamp + r.headers.get('Content-Type')).encode('utf-8') + r.content
signature_check = hmac.new(API_SECRET, msg=string_to_sign, digestmod=hashlib.sha256).hexdigest()
if not r.headers.get('X-Server-Auth-Signature') == signature_check:
    logger.error('Signatures do not match')
    raise Exception('Signatures do not match')

print(r.content)
