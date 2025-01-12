# src/tdr_core/websocket_client.py

import asyncio
import websockets
import json
import logging
from datetime import datetime

###############################################################################
async def subscribe_to_websocket(url: str, symbol: str, data_manager, stop_event):
    """
    Async function to subscribe to the Bitstamp WebSocket for a given symbol,
    storing new trades into data_manager.

    If more than 2 minutes pass with no incoming trades, attempt reconnect.
    """
    # NOTE: We preserve the original references to STALE_FEED_SECONDS by 
    # passing it through if needed. For minimal changes, we keep this as-is
    # but we might need to import or define STALE_FEED_SECONDS locally 
    # if the logic is required here.

    # For demonstration, we keep the same code.
    STALE_FEED_SECONDS = 120
    channel = f"live_trades_{symbol}"

    while not stop_event.is_set():
        last_message_time = datetime.utcnow()
        try:
            data_manager.logger.info(f"{symbol}: Attempting to connect to WebSocket...")
            async with websockets.connect(url) as websocket:
                data_manager.logger.info(f"{symbol}: Connected to WebSocket.")

                subscribe_message = {
                    "event": "bts:subscribe",
                    "data": {"channel": channel}
                }
                await websocket.send(json.dumps(subscribe_message))
                data_manager.logger.info(f"{symbol}: Subscribed to channel: {channel}")

                while not stop_event.is_set():
                    now = datetime.utcnow()
                    seconds_since_last = (now - last_message_time).total_seconds()
                    if seconds_since_last > STALE_FEED_SECONDS:
                        data_manager.logger.warning(
                            f"{symbol}: No trades in {seconds_since_last:.0f} s. Reconnecting..."
                        )
                        break

                    try:
                        # Wait for data up to 10s
                        message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    except asyncio.TimeoutError:
                        continue

                    data_manager.logger.debug(f"{symbol}: {message}")
                    data = json.loads(message)
                    if data.get('event') == 'trade':
                        price = data['data']['price']
                        timestamp = int(float(data['data']['timestamp']))
                        data_manager.add_trade(symbol, price, timestamp, "Live Trade")
                        last_message_time = datetime.utcnow()

        except websockets.ConnectionClosed:
            if stop_event.is_set():
                break
            data_manager.logger.error(f"{symbol}: Connection closed, retrying in 5 seconds...")
            await asyncio.sleep(5)
        except Exception as e:
            if stop_event.is_set():
                break
            data_manager.logger.error(f"{symbol}: An error occurred: {str(e)}")
            await asyncio.sleep(5)
