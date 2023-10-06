import json
import websocket
import os
import logging
import threading
import time

# Configure logging
logging.basicConfig(filename='websocket.log', level=logging.DEBUG)

# Global exit flag
exit_flag = False


def on_message(ws, message):
    global exit_flag
    try:
        logging.debug(f"Received message: {message}")
        data = json.loads(message)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        logging.error(f"Faulty Message: {message}")
        print("An error occurred. Please check the 'websocket.log' for more details.")

        # Write the erroneous message to a file for inspection
        with open("error_message.txt", "w") as error_file:
            error_file.write(message)

        exit_flag = True  # Set the exit flag to True
        return

    # Define the path to save the data
    path = f"data/{currency_pair}_ticker.json"

    # If the file does not exist, create it and initialize with an empty list
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump([], f)

    # Read existing data from file
    with open(path, "r") as f:
        existing_data = json.load(f)

    # Append new data
    existing_data.append(data)

    # Save updated data back to file
    with open(path, "w") as f:
        json.dump(existing_data, f)


def monitor_flag(ws):
    global exit_flag
    while True:
        if exit_flag:
            ws.close()
            break
        time.sleep(1)


def on_error(ws, error):
    print(f"Error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


def on_open(ws):
    # Subscribe to live_trades channel for the currency pair
    payload = {
        "event": "bts:subscribe",
        "data": {
            "channel": f"live_trades_{currency_pair}"
        }
    }
    ws.send(json.dumps(payload))


if __name__ == "__main__":
    # Make sure data directory exists
    if not os.path.exists("data"):
        os.mkdir("data")

    # Define the currency pair
    currency_pair = "btcusd"  # Change this to the currency pair you are interested in

    # Initialize WebSocket connection
    ws = websocket.WebSocketApp(
        "wss://ws.bitstamp.net",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.on_open = on_open
    monitor_thread = threading.Thread(target=monitor_flag, args=(ws,))
    monitor_thread.start()
    ws.run_forever()
