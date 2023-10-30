#!/usr/bin/env python
# src/place-orders.py

import argparse
import hashlib
import hmac
import json
import logging
import time
import uuid
from urllib.parse import urlencode
import requests


def read_config(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def read_order_data(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def create_payload(order):
    order_type = order.get('order_type', 'limit-buy')
    payload = {}
    endpoint = ''

    if order_type in ['limit-buy', 'limit-sell']:
        endpoint = f"/api/v2/{'buy' if order_type == 'limit-buy' else 'sell'}/"
        payload = {'amount': str(order['amount']),
                   'price': str(order['price'])}
    elif order_type in ['market-buy', 'instant-buy', 'market-sell', 'instant-sell']:
        endpoint = f"/api/v2/{'buy' if order_type in ['market-buy', 'instant-buy'] else 'sell'}/market/"
        price_value = str(order.get('price', '28113'))
        payload = {'amount': str(order['amount']), 'price': price_value}
        print(f"Constructed Trade Payload: {json.dumps(payload)}")
    else:
        raise ValueError(f"Unsupported order type: {order_type}")

    return payload, endpoint


def create_message(api_key, endpoint, currency_pair, content_type, nonce, timestamp, payload_string):
    return f"BITSTAMP {api_key}POSTwww.bitstamp.net{endpoint}{currency_pair}/{content_type}{nonce}{timestamp}v2{payload_string}"


def setup_logging():
    logging.basicConfig(filename='bitstamp.log', level=logging.INFO)

def fetch_order_status(api_key, API_SECRET, order_id):
    url = f"https://www.bitstamp.net/api/v2/order_status/"
    nonce = str(uuid.uuid4())
    timestamp = str(int(round(time.time() * 1000)))
    content_type = 'application/x-www-form-urlencoded'
    payload = {'id': order_id}
    message = create_message(api_key, url, "", content_type, nonce, timestamp, urlencode(payload))

    signature = hmac.new(API_SECRET, msg=message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
    headers = {
        'X-Auth': f'BITSTAMP {api_key}',
        'X-Auth-Signature': signature,
        'X-Auth-Nonce': nonce,
        'X-Auth-Timestamp': timestamp,
        'X-Auth-Version': 'v2',
        'Content-Type': content_type
    }

    r = requests.post(url, headers=headers, data=urlencode(payload))

    if r.status_code == 200:
        return json.loads(r.content.decode('utf-8'))
    else:
        print(f"Error fetching order status. Status Code: {r.status_code}. Message: {r.text}")
        return None

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Process Bitstamp orders.')
    parser.add_argument('order_file', type=str, help='Order file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Read configurations
    config = read_config('.bitstamp')
    api_key = config['api_key']
    API_SECRET = bytes(config['api_secret'], 'utf-8')
    order = read_order_data(args.order_file)

    timestamp = str(int(round(time.time() * 1000)))
    nonce = str(uuid.uuid4())
    content_type = 'application/x-www-form-urlencoded'
    payload, endpoint = create_payload(order)
    message = create_message(
        api_key, endpoint, order['currency_pair'], content_type, nonce, timestamp, urlencode(payload))

    signature = hmac.new(API_SECRET, msg=message.encode(
        'utf-8'), digestmod=hashlib.sha256).hexdigest()
    headers = {
        'X-Auth': f'BITSTAMP {api_key}',
        'X-Auth-Signature': signature,
        'X-Auth-Nonce': nonce,
        'X-Auth-Timestamp': timestamp,
        'X-Auth-Version': 'v2',
        'Content-Type': content_type
    }

    setup_logging()

    url = f"https://www.bitstamp.net{endpoint}{order['currency_pair']}/"
    if args.verbose:
        print(f"Request Method: POST")
        print(f"Request URL: {url}")
        print(f"Request Headers: {headers}")
        print(f"Request Payload: {urlencode(payload)}")
    else:
        logging.info(f"Request Method: POST")
        logging.info(f"Request URL: {url}")
        logging.info(f"Request Headers: {headers}")
        logging.info(f"Request Payload: {urlencode(payload)}")

    r = requests.post(url, headers=headers, data=urlencode(payload))

    if r.status_code != 200:
        error_message = f"Error Response from Bitstamp API: {r.text}"
        print(error_message)
        logging.error(f"Unexpected status code: {r.status_code}")
        logging.error(error_message)
        raise Exception('Status code not 200')

    string_to_sign = f"{nonce}{timestamp}{r.headers.get('Content-Type')}".encode(
        'utf-8') + r.content
    signature_check = hmac.new(
        API_SECRET, msg=string_to_sign, digestmod=hashlib.sha256).hexdigest()

    if r.headers.get('X-Server-Auth-Signature') != signature_check:
        logging.error('Signatures do not match')
        raise Exception('Signatures do not match')

    if args.verbose:
        print(f"Response: {r.content}")
    else:
        logging.info(f"Response: {r.content}")

    print(r.content)

    order_response = json.loads(r.content.decode('utf-8'))

    if 'id' in order_response:
        order_id = order_response['id']
        print(f"Fetched order ID: {order_id}")

        for i in range(5):
            time.sleep(1)
            status = fetch_order_status(api_key, API_SECRET, order_id)
            if status:
                print(f"Order status for iteration {i + 1}:")
                print(json.dumps(status, indent=4))
            else:
                print(
                    f"Failed to retrieve order status for iteration {i + 1}.")
    else:
        print("Order ID not found in response. Can't fetch status.")


if __name__ == '__main__':
    main()
