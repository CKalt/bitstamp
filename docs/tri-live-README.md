# Crypto Arbitrage Bot

This is a Python script for a cryptocurrency arbitrage bot that monitors real-time trade data from the Bitstamp exchange for BCH/BTC, BCH/USD, and BTC/USD pairs. It checks for arbitrage opportunities and executes trades when profitable opportunities are found.

## Historical Data

The bot logs historical ticker data to CSV files during the live feed mode. These CSV files are stored in the `historical_data` directory, which is created in the same location as the script.

### File Format

Each currency pair has its own CSV file, named after the pair symbol (e.g., `bchbtc.csv`, `bchusd.csv`, `btcusd.csv`). The CSV files have the following format:

id,timestamp,amount,price,type,microtimestamp,buy_order_id,sell_order_id
1234567890,1622000000,0.12345678,50000.00,0,1622000000000000,1234567890,9876543210
...

The columns in the CSV files are:

- `id`: The unique identifier of the trade.
- `timestamp`: The timestamp of the trade in seconds since the Unix epoch.
- `amount`: The amount of the base currency traded.
- `price`: The price of the trade in the quote currency.
- `type`: The type of the trade (0 for buy, 1 for sell).
- `microtimestamp`: The timestamp of the trade in microseconds since the Unix epoch.
- `buy_order_id`: The identifier of the buy order associated with the trade.
- `sell_order_id`: The identifier of the sell order associated with the trade.

### Processing Historical Data

To run the bot on historical data, use the `--historical` command-line option:

python crypto_arbitrage_bot.py --historical

When running in historical mode, the bot will read the ticker data from the CSV files in the `historical_data` directory and process it as if it were being received in real-time. This allows for testing and refinement of the trading strategy using past data.

## Running the Bot

To run the bot in different modes, use the following command-line options:

- Real-time mode (executes real trades):
  python crypto_arbitrage_bot.py

- Dry-run mode (simulates trades):
  python crypto_arbitrage_bot.py --dry-run

- Historical mode (processes historical data):
  python crypto_arbitrage_bot.py --historical

Note: The `--dry-run` and `--historical` options are mutually exclusive and cannot be used together.

## Configuration

The bot's configuration options, such as the profit threshold, transaction fee, and data freshness thresholds, can be modified in the script's global variables section.

Please ensure that you have the necessary dependencies installed and have set up the required Bitstamp API credentials before running the bot.

## Disclaimer

This bot is provided for educational and informational purposes only. The use of this bot for actual trading is at your own risk. The authors and contributors are not responsible for any losses incurred while using this bot. Please ensure that you understand the risks involved in trading cryptocurrencies before using this bot with real funds.