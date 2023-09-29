import json
import asyncio
import websockets
import time

async def subscribe(url: str, channel: str):
    while True:  # Keep trying to reconnect
        try:
            async with websockets.connect(url) as websocket:
                # Subscribing to the live_trades_btcusd channel.
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }))

                # Receiving messages.
                async for message in websocket:
                    print(message)
                    # Appending messages to the log file.
                    with open("btc-usd.log", "a") as file:
                        file.write(message + "\n")
        except websockets.ConnectionClosed:
            print("Connection closed, trying to reconnect in 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds before trying to reconnect
        except Exception as e:
            print(f"An error occurred: {e}")

# URL for the Bitstamp WebSocket API.
url = 'wss://ws.bitstamp.net'
# Channel for BTC/USD live trades.
channel = 'live_trades_btcusd'

asyncio.get_event_loop().run_until_complete(subscribe(url, channel))
