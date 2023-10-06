import hashlib
import hmac
import time
import requests
import uuid
import sys
import logging

api_key = 'lMf7TEXMUM5x0oWfXp1jrLQOCH6IXv4Z'
API_SECRET = b'374vvoUKJ9YgrxXKWDIGe9pdlSH1gEj5'

timestamp = str(int(round(time.time() * 1000)))
nonce = str(uuid.uuid4())
content_type = 'application/x-www-form-urlencoded'
payload = {'amount': '0.01', 'price': '25300'}

if sys.version_info.major >= 3:
    from urllib.parse import urlencode
else:
    from urllib import urlencode

payload_string = urlencode(payload)

# '' (empty string) in message represents any query parameters or an empty string in case there are none
message = 'BITSTAMP ' + api_key + \
    'POST' + \
    'www.bitstamp.net' + \
    '/api/v2/buy/btcusd/' + \
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
logging.basicConfig(filename='working_code.log', level=logging.INFO)

url = 'https://www.bitstamp.net/api/v2/buy/btcusd/'
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

logging.info(f"Response: {r.content}")
print(r.content)
