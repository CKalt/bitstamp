import asyncio
import websockets
import json
import time
import argparse
import hashlib
import hmac
import uuid
from urllib.parse import urlencode
import requests
from collections import deque
import cmd
import threading

class CryptoDataManager:
    def __init__(self, symbols):
        self.data = {symbol: deque(maxlen=3600) for symbol in symbols}  # Store last hour of data

    def add_trade(self, symbol, price, timestamp):
        self.data[symbol].append((timestamp, float(price)))

    def get_current_price(self, symbol):
        if self.data[symbol]:
            return self.data[symbol][-1][1]
        return None

    def get_price_range(self, symbol, minutes):
        now = time.time()
        relevant_data = [price for ts, price in self.data[symbol] if now - ts <= minutes * 60]
        if relevant_data:
            return min(relevant_data), max(relevant_data)
        return None, None

async def subscribe_to_websocket(url, symbol, data_manager):
    channel = f"live_trades_{symbol}"
    while True:
        try:
            async with websockets.connect(url) as websocket:
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {"channel": channel}
                }))
                async for message in websocket:
                    data = json.loads(message)
                    if data['event'] == 'trade':
                        price = data['data']['price']
                        timestamp = data['data']['timestamp']
                        data_manager.add_trade(symbol, price, timestamp)
        except Exception as e:
            print(f"WebSocket error for {symbol}: {e}")
            await asyncio.sleep(5)

class OrderPlacer:
    def __init__(self, config_file='.bitstamp'):
        self.config = self.read_config(config_file)
        self.api_key = self.config['api_key']
        self.api_secret = bytes(self.config['api_secret'], 'utf-8')

    @staticmethod
    def read_config(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    def place_order(self, order_type, currency_pair, amount, price=None):
        timestamp = str(int(round(time.time() * 1000)))
        nonce = str(uuid.uuid4())
        content_type = 'application/x-www-form-urlencoded'
        
        payload = {'amount': str(amount)}
        if price:
            payload['price'] = str(price)
        
        endpoint = f"/api/v2/{'buy' if 'buy' in order_type else 'sell'}/{'market/' if 'market' in order_type else ''}{currency_pair}/"
        
        message = f"BITSTAMP {self.api_key}POSTwww.bitstamp.net{endpoint}{content_type}{nonce}{timestamp}v2{urlencode(payload)}"
        signature = hmac.new(self.api_secret, msg=message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()
        
        headers = {
            'X-Auth': f'BITSTAMP {self.api_key}',
            'X-Auth-Signature': signature,
            'X-Auth-Nonce': nonce,
            'X-Auth-Timestamp': timestamp,
            'X-Auth-Version': 'v2',
            'Content-Type': content_type
        }
        
        url = f"https://www.bitstamp.net{endpoint}"
        r = requests.post(url, headers=headers, data=urlencode(payload))
        
        if r.status_code == 200:
            return json.loads(r.content.decode('utf-8'))
        else:
            return f"Error: {r.status_code} - {r.text}"

class CryptoShell(cmd.Cmd):
    intro = 'Welcome to the Crypto Shell. Type help or ? to list commands.\n'
    prompt = '(crypto) '

    def __init__(self, data_manager, order_placer):
        super().__init__()
        self.data_manager = data_manager
        self.order_placer = order_placer

    def do_price(self, arg):
        """Show current price for a symbol: price <symbol>"""
        symbol = arg.strip().lower()
        price = self.data_manager.get_current_price(symbol)
        if price:
            print(f"Current price of {symbol}: ${price:.2f}")
        else:
            print(f"No data available for {symbol}")

    def do_range(self, arg):
        """Show price range for last n minutes: range <symbol> <minutes>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: range <symbol> <minutes>")
            return
        symbol, minutes = args[0].lower(), int(args[1])
        min_price, max_price = self.data_manager.get_price_range(symbol, minutes)
        if min_price and max_price:
            print(f"Price range for {symbol} in last {minutes} minutes:")
            print(f"Min: ${min_price:.2f}, Max: ${max_price:.2f}")
        else:
            print(f"No data available for {symbol} in the last {minutes} minutes")

    def do_buy(self, arg):
        """Place a market buy order: buy <symbol> <amount>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: buy <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-buy", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_sell(self, arg):
        """Place a market sell order: sell <symbol> <amount>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: sell <symbol> <amount>")
            return
        symbol, amount = args[0].lower(), float(args[1])
        result = self.order_placer.place_order("market-sell", symbol, amount)
        print(json.dumps(result, indent=2))

    def do_quit(self, arg):
        """Quit the program"""
        print("Quitting...")
        return True

def main():
    symbols = ["btcusd", "ethusd"]  # Add more symbols as needed
    data_manager = CryptoDataManager(symbols)
    order_placer = OrderPlacer()

    # Start WebSocket connections in a separate thread
    def run_websocket():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        url = 'wss://ws.bitstamp.net'
        tasks = [subscribe_to_websocket(url, symbol, data_manager) for symbol in symbols]
        loop.run_until_complete(asyncio.gather(*tasks))

    websocket_thread = threading.Thread(target=run_websocket, daemon=True)
    websocket_thread.start()

    # Start the shell
    CryptoShell(data_manager, order_placer).cmdloop()

if __name__ == '__main__':
    main()