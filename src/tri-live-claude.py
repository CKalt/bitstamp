import json
import os
import asyncio
import websockets
import pandas as pd
import time
import csv

# Constants
HISTORICAL_DATA_DIR = "historical_data"   
URL = 'wss://ws.bitstamp.net'
PROFIT_THRESHOLD = 0.0001
TRANSACTION_FEE = 0.002
DATA_FRESHNESS_THRESHOLD = 60   
SKIP_TRADE_DELAY = 60

# Ensure historical data directory exists
if not os.path.exists(HISTORICAL_DATA_DIR):
    os.makedirs(HISTORICAL_DATA_DIR)

# Global data buffers
data_buffers = {}

# Track last arbitrage timestamp  
last_arbitrage_timestamp = None

def load_symbols():
  with open("websock-ticker-config.json") as f:
    return json.load(f)

async def subscribe(symbol):
    channel = f"live_trades_{symbol}"
    
    while True:
        try:
            async with websockets.connect(URL) as websocket:
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel   
                    }
                }))
                
                async for message in websocket:
                    await handle_message(symbol, message)
                    
        except Exception as e:
            print(f"{symbol}: Connection error: {e}")
            await asyncio.sleep(5)
            
async def handle_message(symbol, message):
    try:
        data = json.loads(message)
        
        if 'data' in data:
            process_trade_data(symbol, data)
        elif data['event'] == 'bts:subscription_succeeded':
            print(f"Subscription succeeded for {symbol}")
        else:
            print(f"Unexpected data for {symbol}: {message}")
            
    except Exception as e:
        print(f"{symbol}: Error processing message: {e}")
        
def process_trade_data(symbol, data):
    timestamp = pd.to_datetime(int(data['data']['timestamp']), unit='s')
    price = data['data']['price']
    
    data_buffers[symbol] = {'timestamp': timestamp, 'price': price}
    
    print(f"Updated {symbol}: {timestamp} with price {price:.8f}") 
    
    log_data(symbol, data)
    check_arbitrage()
    
def log_data(symbol, data):
    log_trade_data(symbol, data)
    log_raw_data(symbol, data)
    
def log_trade_data(symbol, data):
    file_path = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as f:
        fieldnames = ['id', 'timestamp', 'amount', 'price', 'type', 'microtimestamp', 'buy_order_id', 'sell_order_id']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        trade_data = {
          'id': data['data']['id'],
          'timestamp': data['data']['timestamp'],
          'amount': data['data']['amount'],
          'price': data['data']['price'],
          'type': data['data']['type'],
          'microtimestamp': data['data']['microtimestamp'],
          'buy_order_id': data['data']['buy_order_id'],
          'sell_order_id': data['data']['sell_order_id']
        }
        
        writer.writerow(trade_data)
        
def log_raw_data(symbol, data):
    log_file = os.path.join(HISTORICAL_DATA_DIR, f"{symbol}.log")
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(data) + '\n')
        
def check_arbitrage():
    global last_arbitrage_timestamp
    
    # Check buffers, timestamps
    if data_stale() or recently_traded():
        return
        
    # Check prices
    bchbtc = data_buffers['bchbtc']['price']
    bchusd = data_buffers['bchusd']['price'] 
    btcusd = data_buffers['btcusd']['price']
    
    profit = calculate_profit(bchbtc, bchusd, btcusd)
    
    if profit > PROFIT_THRESHOLD:
        print("Arbitrage opportunity found!")
        log_arbitrage_trade(profit)
        
def data_stale():
    current_time = pd.Timestamp.now()
    
    for data in data_buffers.values():
        if (current_time - data['timestamp']).seconds > DATA_FRESHNESS_THRESHOLD:
            return True
            
    return False
    
def recently_traded():
    if last_arbitrage_timestamp:
        time_since_last = pd.Timestamp.now() - last_arbitrage_timestamp
        if time_since_last.seconds < SKIP_TRADE_DELAY:
            return True
            
    return False
    
def calculate_profit(bchbtc, bchusd, btcusd):
    # Calculate profit based on prices
    return profit 

def log_arbitrage_trade(profit):
    last_arbitrage_timestamp = pd.Timestamp.now()
    
    print(f"Logging arbitrage trade at {last_arbitrage_timestamp} for {profit:.8f} BTC")
    
    # Log trade details to file
    
async def main():
    symbols = load_symbols() 
    
    await asyncio.gather(*(subscribe(symbol) for symbol in symbols))
    
if __name__ == "__main__":
    asyncio.run(main())
