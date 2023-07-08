import requests
import hashlib
import hmac
import time
import json
import logging
import uuid

# Load secrets from .bitstamp file
with open('.bitstamp') as f:
    secrets = json.load(f)

api_key = secrets['api_key']
api_secret = secrets['api_secret']

def generate_signature(api_key, api_secret, timestamp, nonce, content_type, payload):
    message = f'BITSTAMP {api_key}POSTwww.bitstamp.net/api/v2/open_orders/all/{content_type}{nonce}{timestamp}v2{payload}'
    signature = hmac.new(api_secret.encode(), msg=message.encode(), digestmod=hashlib.sha256).hexdigest()
    return signature

def list_orders():
    nonce = str(uuid.uuid4())
    timestamp = str(int(time.time() * 1000))
    payload = ''
    content_type = 'application/x-www-form-urlencoded'
    signature = generate_signature(api_key, api_secret, timestamp, nonce, content_type, payload)

    headers = {
        'X-Auth': 'BITSTAMP ' + api_key,
        'X-Auth-Signature': signature,
        'X-Auth-Nonce': nonce,
        'X-Auth-Timestamp': timestamp,
        'X-Auth-Version': 'v2'
    }

    response = requests.post('https://www.bitstamp.net/api/v2/open_orders/all/', headers=headers)
    orders = response.json()
    print(orders)

    # Filter orders for btcusd
    btcusd_orders = [order for order in orders if order['currency_pair'] == 'btcusd']

    return btcusd_orders

def cancel_order(order_id):
    nonce = str(uuid.uuid4())
    timestamp = str(int(time.time() * 1000))
    payload = f'id={order_id}'
    content_type = 'application/x-www-form-urlencoded'
    signature = generate_signature(api_key, api_secret, timestamp, nonce, content_type, payload)

    headers = {
        'X-Auth': 'BITSTAMP ' + api_key,
        'X-Auth-Signature': signature,
        'X-Auth-Nonce': nonce,
        'X-Auth-Timestamp': timestamp,
        'X-Auth-Version': 'v2',
        'Content-Type': content_type
    }

    response = requests.post('https://www.bitstamp.net/api/v2/cancel_order/', headers=headers, data=payload)
    return response.json()

# List open orders
orders = list_orders()

print("Open orders:")
for order in orders:
    print(order)

    # Ask for confirmation to cancel this specific order
    confirm = input(f"Do you want to cancel order {order['id']}? (yes/no): ")
    if confirm.lower() == "yes":
        cancel_order(order['id'])
        print(f"Cancelled order {order['id']}")
