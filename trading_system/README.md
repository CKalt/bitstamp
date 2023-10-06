
# Arbitrage Trading System

The Arbitrage Trading System is a Python-based application designed to identify and execute arbitrage trading opportunities across three currency pairs: BTC/USD, BCH/USD, and BCH/BTC. The system monitors the prices of these currency pairs and executes trades to realize profits from price discrepancies.

## Overview

The trading system continuously monitors the prices of the three currency pairs and calculates potential arbitrage opportunities. The trading is based on two possible cycles:

1. **Cycle 1**: USD -> BTC -> BCH -> USD
2. **Cycle 2**: USD -> BCH -> BTC -> USD

The system evaluates the potential profit from trading in these cycles and executes trades if the profit exceeds a predefined threshold, considering the trading fees.

## How It Works

1. **Price Monitoring**:
   - The system continuously monitors and updates the prices of BTC/USD, BCH/USD, and BCH/BTC.
   
2. **Opportunity Calculation**:
   - It calculates potential arbitrage opportunities based on the current prices of the currency pairs.
   
3. **Trade Execution**:
   - If a profitable opportunity is identified (above a set threshold), the system executes the trades in the respective cycle.
   
4. **Balance Update**:
   - After executing the trades, the system updates the balances of USD, BTC, and BCH.

## Code Structure

1. `trading_system.py`:
   - Contains the main function to load parameters, monitor prices, and make trading decisions.
   
2. `trading_engine.py`:
   - Responsible for making trading decisions based on price data and balance information.
   - Calculates arbitrage opportunities and determines the trading cycle.
   
3. `trade_executor.py`:
   - Responsible for executing the trades based on the identified opportunities and updating the balances.
   
## Example

Based on the recent data analysis, consider the following prices:

- BTC/USD: 50000
- BCH/USD: 500
- BCH/BTC: 0.01

**Cycle 1 (USD -> BTC -> BCH -> USD)**:

1. Start with 1000 USD.
2. Buy BTC for 1000 USD at 50000 USD/BTC, receive 0.02 BTC.
3. Use 0.02 BTC to buy BCH at 0.01 BCH/BTC, receive 2 BCH.
4. Sell 2 BCH for USD at 500 USD/BCH, receive 1000 USD.

In this example, no profit is made, but the system would calculate the potential profit from the trades and execute them if the profit exceeds the set threshold and trading fees.
