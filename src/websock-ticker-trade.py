#! env/bin/python
# src/websock-ticker-trade.py
import json
import asyncio
import websockets
import time

# Handle messages and update the latest prices
async def handle_messages(websocket, symbol, log_filename, parameters):
    async for message in websocket:
        print(f"{symbol}: {message}")
        with open(log_filename, "a") as file:
            file.write(message + "\n")

        data = json.loads(message).get('data', {})
        price = data.get('price')

        if symbol == 'bchbtc':
            parameters['bchbtc_price'] = price
        elif symbol == 'bchusd':
            parameters['bchusd_price'] = price
        elif symbol == 'btcusd':
            parameters['btcusd_price'] = price

# Trading task to execute trades at a certain frequency
async def trading_task(parameters):
    while True:
        last_trade = parameters.get('last_trade', {})
        if last_trade.get('potential_profit', 0) > parameters.get('profit_threshold', 100):
            trading_frequency = parameters.get('increased_frequency', 300)
        else:
            trading_frequency = parameters.get('default_frequency', 600)

        if all(k in parameters for k in ('bchbtc_price', 'bchusd_price', 'btcusd_price')):
            implied_bchusd_price = parameters['btcusd_price'] * parameters['bchbtc_price']
            arbitrage_opportunity = abs(implied_bchusd_price - parameters['bchusd_price'])
            if arbitrage_opportunity > parameters.get('arbitrage_opportunity_threshold', 0.5):
                trade_amount = parameters['trade_amount']
                potential_profit = arbitrage_opportunity * trade_amount
                trade_data = {
                    "timestamp": time.time(),
                    "trade_type": "Buy" if implied_bchusd_price > parameters['bchusd_price'] else "Sell",
                    "potential_profit": potential_profit
                }
                with open("trades.json", "a") as file:
                    file.write(json.dumps(trade_data) + "\n")
                parameters['last_trade'] = trade_data
        await asyncio.sleep(trading_frequency)

# Subscribe to the WebSocket for a given symbol
async def subscribe(url: str, symbol: str, parameters):
    channel = f"live_trades_{symbol}"
    log_filename = f"{symbol}.log"
    while True:
        try:
            async with websockets.connect(url) as websocket:
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }))
                asyncio.create_task(handle_messages(websocket, symbol, log_filename, parameters))
                asyncio.create_task(trading_task(parameters))
        except websockets.ConnectionClosed:
            print(f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"{symbol}: An error occurred: {e}")

# Main function to start the subscription tasks
async def main():
    url = 'wss://ws.bitstamp.net'
    try:
        with open("websock-ticker-config.json", "r") as file:
            symbols = json.load(file)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return
    try:
        with open("parameters.json", "r") as file:
            parameters = json.load(file)
    except Exception as e:
        print(f"Error reading parameters file: {e}")
        return
    await asyncio.gather(*(subscribe(url, symbol, parameters) for symbol in symbols))

# Run the main function
asyncio.get_event_loop().run_until_complete(main())
