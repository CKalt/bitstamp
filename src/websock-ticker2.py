#! env/bin/python
# src/websock-ticker2.py
import json
import asyncio
import websockets
import time

async def subscribe(url: str, symbol: str):
    channel = f"live_trades_{symbol}"
    log_filename = f"{symbol}.log"

    while True:  # Keep trying to reconnect
        try:
            async with websockets.connect(url) as websocket:
                # Subscribing to the channel.
                await websocket.send(json.dumps({
                    "event": "bts:subscribe",
                    "data": {
                        "channel": channel
                    }
                }))

                # Receiving messages.
                async for message in websocket:
                    print(f"{symbol}: {message}")
                    # Appending messages to the log file.
                    with open(log_filename, "a") as file:
                        file.write(message + "\n")
        except websockets.ConnectionClosed:
            print(f"{symbol}: Connection closed, trying to reconnect in 5 seconds...")
            time.sleep(5)  # Wait for 5 seconds before trying to reconnect
        except Exception as e:
            print(f"{symbol}: An error occurred: {e}")

async def main():
    # URL for the Bitstamp WebSocket API.
    url = 'wss://ws.bitstamp.net'

    # Reading the configuration file.
    try:
        with open("websock-ticker-config.json", "r") as file:
            symbols = json.load(file)
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        return

    # Starting subscription tasks.
    await asyncio.gather(*(subscribe(url, symbol) for symbol in symbols))

asyncio.get_event_loop().run_until_complete(main())
