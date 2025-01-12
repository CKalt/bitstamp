# src/tdr_core/order_placer.py

import os
import json
import logging

###############################################################################
class OrderPlacer:
    """
    Handles order placement with the exchange (e.g. Bitstamp).
    """
    def __init__(self, config_file='.bitstamp'):
        self.config_file = config_file
        self.config = self.read_config(self.config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    def read_config(self, file_name):
        file_path = os.path.abspath(file_name)
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read config file '{file_name}': {e}")

    def place_order(self, order_type, currency_pair, amount, price=None, **kwargs):
        import time
        import uuid
        import hmac
        import hashlib
        from urllib.parse import urlencode
        import requests

        timestamp = str(int(round(time.time() * 1000)))
        nonce = str(uuid.uuid4())
        content_type = 'application/x-www-form-urlencoded'

        amount_rounded = round(amount, 8)
        payload = {'amount': str(amount_rounded)}
        if price:
            payload['price'] = str(price)

        for key, value in kwargs.items():
            if value is not None:
                payload[key] = str(value).lower() if isinstance(value, bool) else str(value)

        if 'market' in order_type:
            endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/market/{currency_pair}/"
        else:
            endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/{currency_pair}/"

        payload_string = urlencode(payload)
        message = (
            f"BITSTAMP {self.api_key}"
            f"POSTwww.bitstamp.net{endpoint}{content_type}{nonce}{timestamp}v2{payload_string}"
        )
        signature = hmac.new(self.api_secret, msg=message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        headers = {
            'X-Auth': f'BITSTAMP {self.api_key}',
            'X-Auth-Signature': signature,
            'X-Auth-Nonce': nonce,
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Version': 'v2',
            'Content-Type': content_type
        }

        logging.info(f"Request Method: POST")
        logging.info(f"Request URL: https://www.bitstamp.net{endpoint}")
        logging.info(f"Request Headers: {headers}")
        logging.info(f"Request Payload: {payload_string}")

        url = f"https://www.bitstamp.net{endpoint}"
        r = requests.post(url, headers=headers, data=payload_string)
        if r.status_code == 200:
            return json.loads(r.content.decode('utf-8'))
        else:
            logging.error(f"Error placing order: {r.status_code} - {r.text}")
            return {"status": "error", "reason": r.text, "code": "API_FAILURE"}

    def place_limit_buy_order(self, currency_pair, amount, price, **kwargs):
        return self.place_order('buy', currency_pair, amount, price, **kwargs)

    def place_limit_sell_order(self, currency_pair, amount, price, **kwargs):
        return self.place_order('sell', currency_pair, amount, price, **kwargs)
