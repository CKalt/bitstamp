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
import os

def setup_logging(log_dir, currency_pair, order_type):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # If log_dir is provided, setup file logging
    if log_dir:
        os.makedirs(f"trades/{log_dir}", exist_ok=True)
        log_file = f"trades/{log_dir}/{currency_pair}-{order_type}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    else:
        # Only add StreamHandler to ensure console logging when log_dir is missing
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

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

    if order_type == 'limit-buy':
        endpoint = f"/api/v2/buy/"
        payload = {'amount': str(order['amount']),
                   'price': str(order['price'])}
    elif order_type == 'limit-sell':
        endpoint = f"/api/v2/sell/"
        payload = {'amount': str(order['amount']),
                   'price': str(order['price'])}
    elif order_type == 'market-buy':
        endpoint = f"/api/v2/buy/market/"
        payload = {'amount': str(order['amount'])}
    elif order_type == 'market-sell':
        endpoint = f"/api/v2/sell/market/"
        payload = {'amount': str(order['amount'])}
    elif order_type in ['limit-stop-buy', 'limit-stop-sell']:
        # Use the regular buy or sell endpoint but include the stop price
        endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/{order['currency_pair']}/"
        payload = {
            'amount': str(order['amount']),
            'price': str(order['price']),
            'stop_price': str(order['stop_price'])
        }
    else:
        raise ValueError(f"Unsupported order type: {order_type}")

    logging.info(f"Constructed Trade Payload: {json.dumps(payload)}")
    return payload, endpoint

def create_message(api_key, endpoint, currency_pair, content_type, nonce, timestamp, payload_string):
    return f"BITSTAMP {api_key}POSTwww.bitstamp.net{endpoint}{currency_pair}/{content_type}{nonce}{timestamp}v2{payload_string}"

def fetch_order_status(api_key, API_SECRET, order_id):
    url = f"https://www.bitstamp.net/api/v2/order_status/"
    timestamp = str(int(round(time.time() * 1000)))
    nonce = str(uuid.uuid4())
    content_type = 'application/x-www-form-urlencoded'
    payload = {'id': order_id}

    # Create the payload string
    payload_string = urlencode(payload)

    # Construct the message string according to Bitstamp documentation
    message = (
        'BITSTAMP ' + api_key +
        'POST' +
        'www.bitstamp.net' +
        '/api/v2/order_status/' +
        '' +
        content_type +
        nonce +
        timestamp +
        'v2' +
        payload_string
    )
    message = message.encode('utf-8')

    # Calculate the signature
    signature = hmac.new(API_SECRET, msg=message,
                         digestmod=hashlib.sha256).hexdigest()

    # Set headers for the request
    headers = {
        'X-Auth': 'BITSTAMP ' + api_key,
        'X-Auth-Signature': signature,
        'X-Auth-Nonce': nonce,
        'X-Auth-Timestamp': timestamp,
        'X-Auth-Version': 'v2',
        'Content-Type': content_type
    }

    # Make the request
    r = requests.post(url, headers=headers, data=payload_string)

    if r.status_code == 200:
        return json.loads(r.content.decode('utf-8'))
    else:
        logging.info(
            f"Error fetching order status. Status Code: {r.status_code}. Message: {r.text}")
        return None

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Process Bitstamp orders.')
    parser.add_argument('--order_file', type=str, help='Order file (optional if order details are provided)', default=None)
    parser.add_argument('--order_type', type=str,
                        help='Type of the order e.g. market-buy, market-sell, limit-stop-buy, limit-stop-sell')
    parser.add_argument('--currency_pair', type=str,
                        help='Currency pair e.g. btcusd')
    parser.add_argument('--amount', type=float, help='Amount to order')
    parser.add_argument('--price', type=float,
                        help='Price of the order', default=None)
    parser.add_argument('--stop_price', type=float,
                        help='Stop price for stop-limit orders', default=None)
    parser.add_argument('-v', '--verbose',
                        action='store_true', help='Verbose output')
    parser.add_argument('--log_dir', type=str,
                        help='Directory for log files', default=None)
    args = parser.parse_args()

    # Read configurations
    config = read_config('.bitstamp')
    api_key = config['api_key']
    API_SECRET = bytes(config['api_secret'], 'utf-8')

    # Use command line arguments if provided, else fallback to reading the JSON file
    if args.order_type and args.currency_pair and args.amount:
        order = {
            'order_type': args.order_type,
            'currency_pair': args.currency_pair,
            'amount': args.amount,
        }
        if args.price:
            order['price'] = args.price
        if args.stop_price:
            order['stop_price'] = args.stop_price
    else:
        if args.order_file:
            order = read_order_data(args.order_file)
        else:
            raise ValueError(
                "Either provide order details through command line options or via a JSON file.")

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

    setup_logging(args.log_dir, order['currency_pair'], order['order_type'])

    url = f"https://www.bitstamp.net{endpoint}{order['currency_pair']}/"
    if args.verbose:
        logging.info(f"Request Method: POST")
        logging.info(f"Request URL: {url}")
        logging.info(f"Request Headers: {headers}")
        logging.info(f"Request Payload: {urlencode(payload)}")
    else:
        logging.info(f"Request Method: POST")
        logging.info(f"Request URL: {url}")
        logging.info(f"Request Headers: {headers}")
        logging.info(f"Request Payload: {urlencode(payload)}")

    r = requests.post(url, headers=headers, data=urlencode(payload))

    if r.status_code != 200:
        error_message = f"Error Response from Bitstamp API: {r.text}"
        logging.info(error_message)
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
        logging.info(f"Response: {r.content}")
    else:
        logging.info(f"Response: {r.content}")

    logging.info(r.content)

    order_response = json.loads(r.content.decode('utf-8'))

    if 'id' in order_response:
        order_id = order_response['id']
        logging.info(f"Fetched order ID: {order_id}")

        order_status = None
        iteration = 1
        while order_status != "Finished":
            time.sleep(0.5)  # Wait for 0.5 seconds before each check
            status = fetch_order_status(api_key, API_SECRET, order_id)
            if status:
                logging.info(f"Order status for iteration {iteration}:")
                logging.info(json.dumps(status, indent=4))
                order_status = status.get("status")
            else:
                logging.info(
                    f"Failed to retrieve order status for iteration {iteration}.")

            iteration += 1  # Increment the iteration count
    else:
        logging.info("Order ID not found in response. Can't fetch status.")

if __name__ == '__main__':
    main()
